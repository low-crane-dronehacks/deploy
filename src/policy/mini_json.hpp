// Tiny JSON reader for the policy .npz 'meta' entry (internal; not installed). Numbers are doubles.
#pragma once

#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace policynav {
namespace json {

struct Value {
  enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Value> arr;
  std::map<std::string, Value> obj;

  bool has(const std::string& k) const { return type == kObject && obj.count(k); }
  const Value& at(const std::string& k) const {
    auto it = obj.find(k);
    if (type != kObject || it == obj.end()) throw std::runtime_error("json: missing key '" + k + "'");
    return it->second;
  }
  double number() const {
    if (type != kNumber) throw std::runtime_error("json: not a number");
    return num;
  }
  const std::string& string() const {
    if (type != kString) throw std::runtime_error("json: not a string");
    return str;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}
  Value parse() {
    Value v = value();
    ws();
    if (i_ != s_.size()) err("trailing characters");
    return v;
  }

 private:
  const std::string& s_;
  std::size_t i_ = 0;

  [[noreturn]] void err(const std::string& m) { throw std::runtime_error("json: " + m + " at " + std::to_string(i_)); }
  void ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  bool lit(const char* w) {
    std::size_t n = std::char_traits<char>::length(w);
    if (s_.compare(i_, n, w) == 0) {
      i_ += n;
      return true;
    }
    return false;
  }
  Value value() {
    ws();
    if (i_ >= s_.size()) err("unexpected end");
    Value v;
    const char c = s_[i_];
    if (c == '{') {
      v.type = Value::kObject;
      ++i_;
      ws();
      if (s_[i_] == '}') { ++i_; return v; }
      for (;;) {
        ws();
        if (s_[i_] != '"') err("expected key");
        std::string k = string_lit();
        ws();
        if (s_[i_++] != ':') err("expected ':'");
        v.obj[k] = value();
        ws();
        if (s_[i_] == ',') { ++i_; continue; }
        if (s_[i_] == '}') { ++i_; return v; }
        err("expected ',' or '}'");
      }
    }
    if (c == '[') {
      v.type = Value::kArray;
      ++i_;
      ws();
      if (s_[i_] == ']') { ++i_; return v; }
      for (;;) {
        v.arr.push_back(value());
        ws();
        if (s_[i_] == ',') { ++i_; continue; }
        if (s_[i_] == ']') { ++i_; return v; }
        err("expected ',' or ']'");
      }
    }
    if (c == '"') {
      v.type = Value::kString;
      v.str = string_lit();
      return v;
    }
    if (lit("true")) { v.type = Value::kBool; v.b = true; return v; }
    if (lit("false")) { v.type = Value::kBool; return v; }
    if (lit("null")) return v;
    if (lit("NaN")) { v.type = Value::kNumber; v.num = std::strtod("nan", nullptr); return v; }
    const char* start = s_.c_str() + i_;
    char* end = nullptr;
    v.num = std::strtod(start, &end);
    if (end == start) err("bad value");
    v.type = Value::kNumber;
    i_ += std::size_t(end - start);
    return v;
  }
  std::string string_lit() {
    std::string out;
    ++i_;   // opening quote
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) err("bad escape");
        char e = s_[i_++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': {
            if (i_ + 4 > s_.size()) err("bad \\u escape");
            unsigned cp = std::stoul(s_.substr(i_, 4), nullptr, 16);
            i_ += 4;
            if (cp < 0x80) {
              out += char(cp);
            } else if (cp < 0x800) {
              out += char(0xC0 | (cp >> 6));
              out += char(0x80 | (cp & 0x3F));
            } else {   // surrogate pairs are not combined: meta strings are ASCII in practice
              out += char(0xE0 | (cp >> 12));
              out += char(0x80 | ((cp >> 6) & 0x3F));
              out += char(0x80 | (cp & 0x3F));
            }
            break;
          }
          default: out += e;
        }
      } else {
        out += c;
      }
    }
    if (i_ >= s_.size()) err("unterminated string");
    ++i_;
    return out;
  }
};

inline Value parse(const std::string& s) { return Parser(s).parse(); }

}  // namespace json
}  // namespace policynav
