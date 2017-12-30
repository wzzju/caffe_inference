//
// Created by yuchen on 17-12-30.
//

#ifndef INTELLIPTR_HELPER_H
#define INTELLIPTR_HELPER_H

#include <string>
#include <chrono>
#include <iostream>
#include <vector>

namespace helper {
    template<typename FuncType>
    class InnerScopeExit {
    public:
        InnerScopeExit(const FuncType _func) : func(_func) {}

        ~InnerScopeExit() { if (!dismissed) { func(); }}

    private:
        FuncType func;
        bool dismissed = false;
    };

    template<typename F>
    InnerScopeExit<F> MakeScopeExit(F f) {
        return InnerScopeExit<F>(f);
    };
}
#define DO_STRING_JOIN(arg1, arg2) arg1 ## arg2
#define STRING_JOIN(arg1, arg2) DO_STRING_JOIN(arg1, arg2)
// SCOPEEXIT宏的作用：无论这个宏出现在何处，它总是在退出其所在的scope之后才执行code指定的代码。
#define SCOPEEXIT(code) auto STRING_JOIN(scope_exit_object_, __LINE__) = helper::MakeScopeExit([&](){code;});

// 计算代码在一定scope内执行所耗费的时间
class CostTimeHelper {
public:
    CostTimeHelper(const std::string &_tag) : tag(_tag) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~CostTimeHelper() {
        stop_time = std::chrono::high_resolution_clock::now();
        const auto cost_time = std::chrono::duration_cast<std::chrono::microseconds>(
                stop_time - start_time).count(); //us
        std::cout << tag << " cost time : " << cost_time << " us" << std::endl;
    }

private:
    std::string tag;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;
};

#endif //INTELLIPTR_HELPER_H
