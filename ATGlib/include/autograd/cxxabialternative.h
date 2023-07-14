#if !defined(__CXXABIALTERNATIVE_H__)
#define __CXXABIALTERNATIVE_H__


#if defined ( _MSC_VER )
#include <windows.h> 
#include <dbghelp.h>
#pragma comment(lib,"dbghelp.lib")     
#else
#include <cxxabi.h>
#endif
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
template <typename T> 
char* GetTypeName(const T& v)
{

    char* realname = (char*)malloc(1024 * sizeof(char));
    //if alloc was OK then set 0 first char and use UnDecorateSymbolName
    realname ? realname[0] = 0, ::UnDecorateSymbolName(typeid(v).name(), realname, 1024, 0) : 0;

    return realname;
    //总是在外部手动释放realname
}

std::string join(const std::vector<std::string>& input, const std::string& separator) {
    // If the input is empty, return an empty string
    if (input.empty()) {
        return "";
    }
    // Initialize the result with the first element of the input
    std::string result = input[0];
    // Loop through the rest of the input and append each element with the separator
    for (size_t i = 1; i < input.size(); i++) {
        result.append(separator);
        result.append(input[i]);
    }
    // Return the result
    return result;
}

#endif
