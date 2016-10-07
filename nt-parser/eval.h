#ifndef NTPARSER_EVAL_H_
#define NTPARSER_EVAL_H_

#include <string>
#include <vector>

namespace parser {

struct EvalBResults {
  float p,r,f;
};

EvalBResults Evaluate(const std::string& ref_fname, const std::string& hyp_fname);

}  // namespace parser

#endif
