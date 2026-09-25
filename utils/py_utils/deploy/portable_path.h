#ifndef BST_PORTABLE_PATH_H
#define BST_PORTABLE_PATH_H

#include <cerrno>
#include <cstddef>
#include <string>
#include <sys/stat.h>

namespace bst_path {

inline std::string parent(const std::string& path) {
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos) return ".";
    return slash == 0 ? "/" : path.substr(0, slash);
}

inline std::string stem(const std::string& path) {
    const auto slash = path.find_last_of('/');
    const std::string name = path.substr(slash == std::string::npos ? 0 : slash + 1);
    const auto dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

inline std::string join(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir == ".") return name;
    return dir == "/" ? dir + name : dir + "/" + name;
}

inline bool ensure_dir(const std::string& path) {
    if (path.empty()) return false;
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i != path.size() && path[i] != '/') continue;
        const std::string part = path.substr(0, i);
        if (part == "." || part.empty()) continue;
        if (mkdir(part.c_str(), 0775) != 0 && errno != EEXIST) return false;
    }
    struct stat info {};
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

}  // namespace bst_path

#endif
