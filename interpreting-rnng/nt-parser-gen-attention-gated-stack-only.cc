#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/functional/hash.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/cfsm-builder.h"

#include "nt-parser/oracle.h"
#include "nt-parser/pretrained.h"
#include "nt-parser/compressed-fstream.h"
#include "nt-parser/eval.h"

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict;

volatile bool requested_stop = false;
unsigned kSOS = 0;
unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;

unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

ClassFactoredSoftmaxBuilder *cfsm = nullptr;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("explicit_terminal_reduce,x", "[not recommended] If set, the parser must explicitly process a REDUCE operation to complete a preterminal constituent")
        ("dropout,D", po::value<float>(), "Use dropout")
        ("clusters,c", po::value<string>(), "Clusters word clusters file")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("eta_decay,e", po::value<float>(), "Start decaying eta after this many epochs")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct ParserBuilder {
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder term_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_nt; // nonterminal embeddings
  LookupParameters* p_nt_query; // nonterminal embeddings
  LookupParameters* p_ntup; // nonterminal embeddings when used in a composed representation
  LookupParameters* p_a; // input action embeddings
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_T; // term lstm to parser state
  //Parameters* p_pbias2; // parser state bias
  //Parameters* p_A2; // action lstm to parser state
  //Parameters* p_S2; // stack lstm to parser state
  //Parameters* p_w2l; // word to LSTM input
  //Parameters* p_t2l; // pretrained word embeddings to LSTM input
  //Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_stack_guard;  // end of stack
  //Add parameter for attention
  Parameters* p_V;
  Parameters* p_V_bias;
  Parameters* p_gate_NT;
  Parameters* p_gate_composed;
  Parameters* p_gate_bias;

  Parameters* p_cW;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      term_lstm(LAYERS, INPUT_DIM, HIDDEN_DIM, model),  // sequence of generated terminals
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      const_lstm_fwd(1, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      const_lstm_rev(1, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_nt(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_nt_query(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_ntup(model->add_lookup_parameters(NT_SIZE, {2 * LSTM_INPUT_DIM})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_T(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      //p_pbias2(model->add_parameters({HIDDEN_DIM})),
      //p_A2(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      //p_S2(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      //p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      //p_ib(model->add_parameters({LSTM_INPUT_DIM})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM})),
      p_abias(model->add_parameters({ACTION_SIZE})),

      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})),
      //Add parameters for attention
      p_V(model->add_parameters({2 * LSTM_INPUT_DIM, HIDDEN_DIM + LSTM_INPUT_DIM})),
      p_V_bias(model->add_parameters({2 * LSTM_INPUT_DIM})),
      p_gate_NT(model->add_parameters({2 * LSTM_INPUT_DIM, 2 * LSTM_INPUT_DIM})),
      p_gate_composed(model->add_parameters({2 * LSTM_INPUT_DIM, 2 * LSTM_INPUT_DIM})),
      p_gate_bias(model->add_parameters({2 * LSTM_INPUT_DIM})),

      p_cW(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM * 2})) {
    if (pretrained.size() > 0) {
      cerr << "Pretrained embeddings not implemented\n";
      abort();
    }
  }

// checks to see if a proposed action is valid in discriminative models
static bool IsActionForbidden_Generative(const string& a, char prev_a, unsigned tsize, unsigned ssize, unsigned nopen_parens) {
  bool is_shift = (a[0] == 'S' && a[1]=='H');
  bool is_reduce = (a[0] == 'R' && a[1]=='E');
  bool is_nt = (a[0] == 'N');
  assert(is_shift || is_reduce || is_nt);
  static const unsigned MAX_OPEN_NTS = 100;
  if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
  if (ssize == 1) {
    if (!is_nt) return true;
    return false;
  }
  // you can't reduce after an NT action
  if (is_reduce && prev_a == 'N') return true;
  return false;
}

// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// if sent is empty, generate a sentence
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
                     bool is_evaluation) {
    vector<unsigned> results;
    vector<string> stack_content;
    stack_content.push_back("ROOT_GUARD");
    const bool sample = sent.size() == 0;
    const bool build_training_graph = correct_actions.size() > 0;
    assert(sample || build_training_graph);
    bool apply_dropout = (DROPOUT && !is_evaluation);
    if (sample) apply_dropout = false;

    if (apply_dropout) {
      stack_lstm.set_dropout(DROPOUT);
      term_lstm.set_dropout(DROPOUT);
      action_lstm.set_dropout(DROPOUT);
      const_lstm_fwd.set_dropout(DROPOUT);
      const_lstm_rev.set_dropout(DROPOUT);
    } else {
      stack_lstm.disable_dropout();
      term_lstm.disable_dropout();
      action_lstm.disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }
    term_lstm.new_graph(*hg);
    stack_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    cfsm->new_graph(*hg);
    term_lstm.start_new_sequence();
    stack_lstm.start_new_sequence();
    action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression A = parameter(*hg, p_A);
    Expression T = parameter(*hg, p_T);
    //Expression pbias2 = parameter(*hg, p_pbias2);
    //Expression S2 = parameter(*hg, p_S2);
    //Expression A2 = parameter(*hg, p_A2);

    //Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    //Expression w2l = parameter(*hg, p_w2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);
    Expression V = parameter(*hg, p_V);
    Expression V_bias = parameter(*hg, p_V_bias);

    action_lstm.add_input(action_start);

    vector<Expression> terms(1, lookup(*hg, p_w, kSOS));
    term_lstm.add_input(terms.back());

    vector<Expression> stack;  // variables representing subtree embeddings
    stack.push_back(parameter(*hg, p_stack_guard));
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    int nopen_parens = 0;
    char prev_a = '0';
    unsigned termc = 0;
    while(stack.size() > 2 || termc == 0) {
      assert (stack.size() == stack_content.size());
      // get list of possible actions for the current parser state
      current_valid_actions.clear();
      for (auto a: possible_actions) {
        if (IsActionForbidden_Generative(adict.Convert(a), prev_a, terms.size(), stack.size(), nopen_parens))
          continue;
        current_valid_actions.push_back(a);
      }
      //cerr << "valid actions = " << current_valid_actions.size() << endl;

      //onerep
      Expression stack_summary = stack_lstm.back();
      Expression action_summary = action_lstm.back();
      Expression term_summary = term_lstm.back();
      if (apply_dropout) {
        stack_summary = dropout(stack_summary, DROPOUT);
        action_summary = dropout(action_summary, DROPOUT);
        term_summary = dropout(term_summary, DROPOUT);
      }
      Expression p_t = affine_transform({pbias, S, stack_summary});
      Expression nlp_t = rectify(p_t);
      //tworep*
      //Expression p_t = affine_transform({pbias, S, stack_lstm.back(), A, action_lstm.back()});
      //Expression nlp_t = rectify(p_t);
      //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});

      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      unsigned action = 0;
      if (sample) {
        auto dist = as_vector(hg->incremental_forward());
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(dist[current_valid_actions[w]]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        action = current_valid_actions[w];
        const string& a = adict.Convert(action);
        if (a[0] == 'R') cerr << ")";
        if (a[0] == 'N') {
          int nt = action2NTindex[action];
          cerr << " (" << ntermdict.Convert(nt);
        }
      } else {
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
        //cerr << endl;
        ++action_count;
        log_probs.push_back(pick(adiste, action));
      }
      results.push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      // do action
      const string& actionString=adict.Convert(action);
      //cerr << "ACT: " << actionString << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];
      prev_a = ac;

      if (ac =='S' && ac2=='H') {  // SHIFT
        unsigned wordid = 0;
        //tworep
        //Expression p_t = affine_transform({pbias2, S2, stack_lstm.back(), A2, action_lstm.back(), T, term_lstm.back()});
        //Expression nlp_t = rectify(p_t);
        //tworep-oneact:
        //Expression p_t = affine_transform({pbias2, S2, stack_lstm.back(), T, term_lstm.back()});
        //Expression nlp_t = rectify(p_t);
        if (sample) {
          wordid = cfsm->sample(nlp_t);
          cerr << " " << termdict.Convert(wordid);
        } else {
          assert(termc < sent.size());
          wordid = sent.raw[termc];
          log_probs.push_back(-cfsm->neg_log_softmax(nlp_t, wordid));
        }
        assert (wordid != 0);
        stack_content.push_back(termdict.Convert(wordid)); //add the string of the word to the stack
        ++termc;
        Expression word = lookup(*hg, p_w, wordid);
        terms.push_back(word);
        term_lstm.add_input(word);
        stack.push_back(word);
        stack_lstm.add_input(word);
        is_open_paren.push_back(-1);
      } else if (ac == 'N') { // NT
        ++nopen_parens;
        auto it = action2NTindex.find(action);
        assert(it != action2NTindex.end());
        int nt_index = it->second;
        nt_count++;
        stack_content.push_back(ntermdict.Convert(nt_index));
        Expression nt_embedding = lookup(*hg, p_nt, nt_index);
        stack.push_back(nt_embedding);
        stack_lstm.add_input(nt_embedding);
        is_open_paren.push_back(nt_index);
      } else { // REDUCE
        --nopen_parens;
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(stack_content.size() > 2 && stack.size() == stack_content.size());
        // find what paren we are closing
        int i = is_open_paren.size() - 1; //get the last thing on the stack
        while(is_open_paren[i] < 0) { --i; assert(i >= 0); } //iteratively decide whether or not it's a non-terminal
        int nt_idx = is_open_paren[i];
        //cerr << "nt_idx" << nt_idx << endl;
        //Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
        int nchildren = is_open_paren.size() - i - 1;
        assert(nchildren > 0);
        //cerr << "  number of children to reduce: " << nchildren << endl;
        vector<Expression> children(nchildren);
        const_lstm_fwd.start_new_sequence();
        const_lstm_rev.start_new_sequence();

        // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
        // TO BE COMPOSED INTO A TREE EMBEDDING
        string curr_word;
        //cerr << "--------------------------------" << endl;
        //cerr << "Now printing the children" << endl;
        //cerr << "--------------------------------" << endl;
        vector<string> to_be_composed(nchildren);
        for (i = nchildren - 1; i >= 0; --i) {
          assert (stack_content.size() == stack.size());
          children[i] = stack.back();
          to_be_composed[i] = stack_content.back();
          stack.pop_back();
          stack_lstm.rewind_one_step();
          is_open_paren.pop_back();
          curr_word = stack_content.back();
          //cerr << "At the back of the stack (supposed to be one of the children): " << curr_word << endl;
          stack_content.pop_back();
        }
        assert (stack_content.size() == stack.size());
        //cerr << "Doing REDUCE operation" << endl;
        // Now pop the (single) non-terminal symbol
        is_open_paren.pop_back(); // nt symbol
        stack.pop_back(); // nonterminal dummy
        stack_lstm.rewind_one_step(); // nt symbol
        curr_word = stack_content.back();
        //cerr << "--------------------------------" << endl;
        //cerr << "At the back of the stack (supposed to be the non-terminal symbol) : " << curr_word << endl;
        stack_content.pop_back();
        assert (stack.size() == stack_content.size());
        //cerr << "Done reducing" << endl;

        // BUILD TREE EMBEDDING USING BIDIR LSTM, AUGMENTED WITH ATTENTION attention
        //cerr << "COMPOSING" << endl;
        //cerr << "--------------------------------" << endl;
        vector<Expression> forward;
        deque<Expression> backward;
        //const_lstm_fwd.add_input(nonterminal);
        //forward.push_back(const_lstm_fwd.back());
        //cerr << "First word (supposed to be the NT): " << curr_word << endl;
        //const_lstm_rev.add_input(nonterminal);
        for (i = 0; i < nchildren; ++i) {
          const_lstm_fwd.add_input(children[i]);
          forward.push_back(const_lstm_fwd.back());
          //cerr << "Forward word: " << to_be_composed[i] << endl;
          //cerr << "backward word: " << to_be_composed[nchildren - i - 1] << endl;
          const_lstm_rev.add_input(children[nchildren - i - 1]);
          backward.push_front(const_lstm_rev.back());
        }
        //const_lstm_rev.add_input(nonterminal);
        //backward.push_front(const_lstm_rev.back());
        assert (forward.size() == backward.size() && forward.size() == (nchildren));
        vector<Expression> concatenated;
        //cerr << "END COMPOSING" << endl;
        //cerr << "--------------------------------" << endl;
        for (unsigned i = 0; i < forward.size(); ++i) {
	  vector<Expression> temp;
          temp.push_back(forward[i]);
          temp.push_back(backward[i]);
          assert (temp.size() == 2);
          concatenated.push_back(concatenate(temp));
        } 
        assert (concatenated.size() == forward.size() && forward.size() == backward.size() && concatenated.size() == (nchildren));
        Expression matrix_sent = concatenate_cols(concatenated);
        auto matrix_eigen = *(hg->incremental_forward());
        //cerr << "Number of cols: " << matrix_eigen.cols() << " and number of rows: " << matrix_eigen.rows() << endl; 
        assert (matrix_eigen.cols() == (nchildren) && matrix_eigen.rows() == (2 * LSTM_INPUT_DIM));
        //cerr << "Before" << endl;
        Expression nt_query_key = lookup(*hg, p_nt_query, nt_idx);
        //auto nt_query_temp = as_vector(hg->incremental_forward());
        //cerr << "After" << endl;
        vector<Expression> nt_query_vector;
        nt_query_vector.push_back(nt_query_key);
        nt_query_vector.push_back(nlp_t);
        Expression nt_query = concatenate(nt_query_vector);
        Expression rt = affine_transform({V_bias, V, nt_query}); 
        Expression ut = transpose(matrix_sent) * rt;
        Expression at = softmax(ut);
        //cerr << "Test 1" << endl;
        // Do some sanity check on the attention vector. Make sure that: 1. The dimensions are right 2. The attention vector sums to one
        auto at_vector = as_vector(hg->incremental_forward());
        assert (at_vector.size() == (nchildren));
        //cerr << "Test 2" << endl;
        float sum = accumulate(at_vector.begin(), at_vector.end(), 0.0f);
        assert (sum >= 0.9999 && sum <= 1.0001);
        // End of the sanity check
        //cerr << "Before ntup" << endl;
        Expression nonterminal = lookup(*hg, p_ntup, nt_idx);
        //cerr << "After ntup" << endl;
        if (apply_dropout) {
          nonterminal = dropout(nonterminal, DROPOUT);
        }
        Expression ct = matrix_sent * at; // Dimension of ct is 2 * LSTM_INPUT_DIM
        if (apply_dropout) {
          ct = dropout(ct, DROPOUT);
        }
        Expression gate_NT = parameter(*hg, p_gate_NT);
        Expression gate_composed = parameter(*hg, p_gate_composed);
        Expression gate_bias = parameter(*hg, p_gate_bias);
        Expression gate = logistic(affine_transform({gate_bias, gate_NT, nonterminal, gate_composed, ct}));
        //cerr << "gate: " << endl;
        //auto gate_vector_temp = *(hg->incremental_forward());
        //cerr << gate_vector_temp << endl;
        //cerr << "--------------------------------------------------" << endl;
        vector<float> ones_vec(2 * LSTM_INPUT_DIM, 1.0f);
        Expression ones = input(*hg, {2 * LSTM_INPUT_DIM, 1}, ones_vec);
        //cerr << "1 - gate: " << endl;
        //Expression ones_gate = ones - gate; 
        //auto one_minus_gate = *(hg->incremental_forward());
        //cerr << one_minus_gate << endl;
        //cerr << "--------------------------------------------------" << endl;
        Expression gated = cwise_multiply(gate, nonterminal) + cwise_multiply(ones - gate, ct); 
        //OLD CODE
        /*
        Expression cfwd = const_lstm_fwd.back();
        Expression crev = const_lstm_rev.back();
        if (apply_dropout) {
          cfwd = dropout(cfwd, DROPOUT);
          crev = dropout(crev, DROPOUT);
        }
        Expression c = concatenate({cfwd, crev}); */
        Expression composed = rectify(affine_transform({cbias, cW, gated}));
        stack_lstm.add_input(composed);
        stack.push_back(composed);
        stack_content.push_back(curr_word);
        //cerr << curr_word << endl;
        is_open_paren.push_back(-1); // we just closed a paren at this position
      }  //END REDUCE
    }
    if (action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }
    assert(stack.size() == 2); // guard symbol, root
    if (!sample) {
      Expression tot_neglogprob = -sum(log_probs);
      assert(tot_neglogprob.pg != nullptr);
    }
    if (sample) cerr << "\n";
    return results;
  }
};

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  if (conf.count("clusters") == 0) {
    cerr << "Please specify vocabulary clustering with --clusters FILE when training generative model\n";
    return 1;
  }
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  if (conf.count("train") && conf.count("dev_data") == 0) {
    cerr << "You specified --train but did not specify --dev_data FILE\n";
    return 1;
  }
  ostringstream os;
  os << "ntparse_gen"
     << "_D" << DROPOUT
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  bool softlinkCreated = false;

  kSOS = termdict.Convert("<s>");
  Model model;
  cfsm = new ClassFactoredSoftmaxBuilder(HIDDEN_DIM, conf["clusters"].as<string>(), &termdict, &model);

  parser::TopDownOracleGen corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::TopDownOracleGen dev_corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::TopDownOracleGen2 test_corpus(&termdict, &adict, &posdict, &ntermdict);
  corpus.load_oracle(conf["training_data"].as<string>());

  if (conf.count("words"))
    parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.Freeze();
  adict.Freeze();
  ntermdict.Freeze();
  posdict.Freeze();

  if (conf.count("dev_data")) {
    cerr << "Loading validation set\n";
    dev_corpus.load_oracle(conf["dev_data"].as<string>());
  }

  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(conf["test_data"].as<string>());
  }

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.Convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.Convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
  }

  NT_SIZE = ntermdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (unsigned i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  ParserBuilder parser(&model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    //AdamTrainer sgd(&model);
    SimpleSGDTrainer sgd(&model);
    //sgd.eta = 0.01;
    //sgd.eta0 = 0.01;
    //MomentumSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08;
    //sgd.eta_decay = 0.05;
    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min((int)status_every_i_iterations, (int)corpus.sents.size());
    unsigned si = corpus.sents.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sents.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev_llh = 9e99;
    //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.sents.size()) {
             si = 0;
             if (first) { first = false; } else {
                  sgd.update_epoch();
                  //sgd.eta /= 2;
              }
             //cerr << "NO SHUFFLE" << endl;
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
             //sgd.eta /= 2;
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<int>& actions=corpus.actions[order[si]];
           ComputationGraph hg;
           parser.log_prob_parser(&hg,sentence,actions,&right,false);
           double lp = as_scalar(hg.incremental_forward());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
           words += sentence.size();
      }
      sgd.status();
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<
         /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
        ") per-action-ppl: " << exp(llh / trs) << " per-input-ppl: " << exp(llh / words) << " per-sent-ppl: " << exp(llh / status_every_i_iterations) << " err: " << (trs - right) / trs << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;
      llh = trs = right = words = 0;
      static int logc = 0;
      ++logc;
      if (logc > 50) {
        // generate random sample
        ComputationGraph cg;
        double x;
        parser.log_prob_parser(&cg, parser::Sentence(), vector<int>(),&x,true);
      }
      if (logc % 100 == 0) { // report on dev set
        unsigned dev_size = dev_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<int>& actions=dev_corpus.actions[sii];
           dwords += sentence.size();
           {  ComputationGraph hg;
              parser.log_prob_parser(&hg,sentence,actions,&right,true);
              double lp = as_scalar(hg.incremental_forward());
              llh += lp;
           }
           double lp = 0;
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        double err = (trs - right) / trs;
        //parser::EvalBResults res = parser::Evaluate("foo", pfx);
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
        if (llh < best_dev_llh && (tot_seen / corpus.size()) > 1.0) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_llh = llh;
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        }
      }
    }
  } // should do training?
  if (test_corpus.size() > 0) {
    // if rescoring, we may have many repeats, cache them
    unordered_map<vector<int>, unordered_map<vector<int>,double,boost::hash<vector<int>>>, boost::hash<vector<int>>> s2a2p;
    unsigned test_size = test_corpus.size();
    double llh = 0;
    double right = 0;
    double dwords = 0;
    for (unsigned sii = 0; sii < test_size; ++sii) {
      const auto& sentence=test_corpus.sents[sii];
      const vector<int>& actions=test_corpus.actions[sii];
      dwords += sentence.size();
      double& lp = s2a2p[sentence.raw][actions];
      if (!lp) {
        ComputationGraph hg;
        parser.log_prob_parser(&hg,sentence,actions,&right,true);
        lp = as_scalar(hg.incremental_forward());
      }
      cout << sentence.size() << '\t' << lp << endl;
      llh += lp;
    }
    cerr << "test     total -llh=" << llh << endl;
    cerr << "test ppl (per word)=" << exp(llh / dwords) << endl;
  }
}
