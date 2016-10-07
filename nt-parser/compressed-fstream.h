#ifndef PARSER_COMPRESSED_FSTREAM_H_
#define PARSER_COMPRESSED_FSTREAM_H_

#include <iostream>
#include <string>
#include <fstream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

namespace cnn {

// acts just like std::ifstream, but decompresses if the filename ends in .gz or .bz2
class compressed_ifstream : public std::istream {
 private:
  std::ifstream file;
  boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
 public:
  compressed_ifstream(const std::string& fname) : std::istream(&inbuf), file(fname.c_str()) {
    std::size_t pos = fname.rfind('.');
    if (pos != std::string::npos && pos > 0) {
      const std::string suf = fname.substr(pos + 1);
      if (suf == "gz") {
        inbuf.push(boost::iostreams::gzip_decompressor());
      }
      else if (suf == "bz2")
        inbuf.push(boost::iostreams::bzip2_decompressor());
    }
    inbuf.push(file);
  }
};

}; // namespace cnn

#endif
