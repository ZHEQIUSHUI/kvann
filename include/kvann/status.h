#pragma once

#include <string>
#include <utility>

namespace kvann {

enum class StatusCode {
    kOk = 0,
    kNotFound,
    kAlreadyExists,
    kDimMismatch,
    kFull,
    kIo,
    kInvalidArgument,
    kUnsupported,
    kInternal,
};

class Status {
public:
    Status() = default;
    Status(StatusCode c, std::string msg = {}) noexcept
        : code_(c), msg_(std::move(msg)) {}

    static Status Ok()                                   { return {}; }
    static Status NotFound(std::string m = {})           { return {StatusCode::kNotFound,        std::move(m)}; }
    static Status AlreadyExists(std::string m = {})      { return {StatusCode::kAlreadyExists,   std::move(m)}; }
    static Status DimMismatch(std::string m = {})        { return {StatusCode::kDimMismatch,     std::move(m)}; }
    static Status Full(std::string m = {})               { return {StatusCode::kFull,            std::move(m)}; }
    static Status Io(std::string m = {})                 { return {StatusCode::kIo,              std::move(m)}; }
    static Status InvalidArgument(std::string m = {})    { return {StatusCode::kInvalidArgument, std::move(m)}; }
    static Status Unsupported(std::string m = {})        { return {StatusCode::kUnsupported,     std::move(m)}; }
    static Status Internal(std::string m = {})           { return {StatusCode::kInternal,        std::move(m)}; }

    bool ok() const noexcept              { return code_ == StatusCode::kOk; }
    StatusCode code() const noexcept      { return code_; }
    const std::string& message() const&   { return msg_; }

    explicit operator bool() const noexcept { return ok(); }

    const char* code_str() const noexcept {
        switch (code_) {
            case StatusCode::kOk:              return "ok";
            case StatusCode::kNotFound:        return "not_found";
            case StatusCode::kAlreadyExists:   return "already_exists";
            case StatusCode::kDimMismatch:     return "dim_mismatch";
            case StatusCode::kFull:            return "full";
            case StatusCode::kIo:              return "io";
            case StatusCode::kInvalidArgument: return "invalid_argument";
            case StatusCode::kUnsupported:     return "unsupported";
            case StatusCode::kInternal:        return "internal";
        }
        return "unknown";
    }

private:
    StatusCode code_ = StatusCode::kOk;
    std::string msg_;
};

} // namespace kvann
