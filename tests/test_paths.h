// Cross-platform temp file path helper for kvann tests.
//
// Linux / macOS: /tmp
// Windows      : %TEMP% (e.g. C:\Users\runneradmin\AppData\Local\Temp)
#pragma once

#include <filesystem>
#include <string>

namespace kvann_test {

inline std::string tmp_path(const std::string& name) {
    std::error_code ec;
    auto p = std::filesystem::temp_directory_path(ec);
    if (ec) {
        // Fallback to CWD if the system has no temp dir resolvable.
        return name;
    }
    return (p / name).string();
}

} // namespace kvann_test
