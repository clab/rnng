#include "nt-parser/pretrained.h"

#include <sstream>
#include "cnn/dict.h"
#include "nt-parser/compressed-fstream.h"

using namespace std;
using namespace cnn;

namespace parser {

void ReadEmbeddings_word2vec(const string& fname,
        Dict* dict,
        unordered_map<unsigned, vector<float>>* pretrained) {
  cerr << "Reading pretrained embeddings from " << fname << " ...\n";
  compressed_ifstream in(fname);
  string line;
  getline(in, line);
  bool bad = false;
  int spaces = 0;
  for (auto c : line) {
    if (c == ' ' || c == '\t') ++spaces;
    else if (c < '0' || c > '9') bad = true;
  }
  if (spaces != 1 || bad) {
    cerr << "File does not seem to be in word2vec format\n";
    abort();
  }
  istringstream iss(line);
  unsigned nwords = 0, dims = 0;
  iss >> nwords >> dims;
  cerr << "    file reports " << nwords << " words with " << dims << " dims\n";
  unsigned lc = 1;
  string word;
  while(getline(in, line)) {
    ++lc;
    vector<float> v(dims);
    istringstream iss(line);
    iss >> word;
    unsigned wordid = dict->Convert(word);
    for (unsigned i = 0; i < dims; ++i)
      iss >> v[i];
    (*pretrained)[wordid] = v;
  }
  if ((lc-1) != nwords) {
    cerr << "[WARNING] mismatched number of words reported and loaded\n";
  }
  cerr << "    done.\n";
}

} // namespace parser
