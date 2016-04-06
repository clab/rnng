#include "nt-parser/eval.h"

#include <iostream>
#include <cstdio>

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

using namespace std;
namespace io = boost::iostreams;

namespace parser {

EvalBResults Evaluate(const string& ref_fname, const string& hyp_fname) {
  string cmd = "echo " + ref_fname + " " + hyp_fname;
  cerr << "COMMAND: " << cmd << endl;
  FILE *pipe = popen(cmd.c_str(), "r");
  io::stream_buffer<io::file_descriptor_source> fpstream (fileno(pipe), io::never_close_handle);
  istream in(&fpstream);
  string line;
  while(getline(in, line)) {
    cerr << "Got line: " << line << endl;
  }
  fclose(pipe);
  EvalBResults r;
  return r;
}

};

