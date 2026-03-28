#pragma once


#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>

namespace nnc_test {

// ---- Счётчики ----
inline int& passed() { static int n = 0; return n; }
inline int& failed() { static int n = 0; return n; }

// ---- Текущий тест для вывода ----
inline std::string& currentTest() { static std::string s; return s; }

// ---- Регистр тестов ----
struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> v;
    return v;
}

struct Registrar {
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        registry().push_back({suite, name, std::move(fn)});
    }
};

// ---- Вывод результата одной проверки ----
inline void checkImpl(bool cond, const char* expr,
                      const std::string& detail,
                      const char* file, int line) {
    if (cond) {
        ++passed();
    } else {
        ++failed();
        std::cerr << "  FAIL " << file << ":" << line
                  << "  [" << expr << "]";
        if (!detail.empty()) std::cerr << "  →  " << detail;
        std::cerr << "\n";
    }
}

// ---- Запуск всех тестов ----
inline int runAll() {
    int total = static_cast<int>(registry().size());
    std::cout << "Running " << total << " tests...\n\n";

    std::string lastSuite;
    for (auto& tc : registry()) {
        if (tc.suite != lastSuite) {
            std::cout << "[ " << tc.suite << " ]\n";
            lastSuite = tc.suite;
        }
        currentTest() = tc.suite + "::" + tc.name;
        std::cout << "  " << tc.name << " ... ";
        std::cout.flush();

        try {
            int before = failed();
            tc.fn();
            if (failed() == before)
                std::cout << "ok\n";
            else
                std::cout << "FAILED\n";
        } catch (const std::exception& e) {
            ++failed();
            std::cout << "EXCEPTION: " << e.what() << "\n";
        } catch (...) {
            ++failed();
            std::cout << "UNKNOWN EXCEPTION\n";
        }
    }

    std::cout << "\n============================\n";
    std::cout << "  Passed : " << passed() << "\n";
    std::cout << "  Failed : " << failed() << "\n";
    std::cout << "============================\n";
    return failed() == 0 ? 0 : 1;
}

} // namespace nnc_test

// ---- Макросы ----

#define CHECK(cond) \
    nnc_test::checkImpl(!!(cond), #cond, "", __FILE__, __LINE__)

#define CHECK_EQ(a, b) \
    do { \
        auto _a = (a); auto _b = (b); \
        nnc_test::checkImpl(_a == _b, #a " == " #b, \
            (_a == _b) ? "" : "values differ", __FILE__, __LINE__); \
    } while(0)

#define CHECK_NE(a, b) \
    do { \
        auto _a = (a); auto _b = (b); \
        nnc_test::checkImpl(!(_a == _b), #a " != " #b, \
            (!(_a == _b)) ? "" : "values are equal", __FILE__, __LINE__); \
    } while(0)

#define CHECK_CONTAINS(str_expr, sub) \
    do { \
        std::string _s = (str_expr); \
        std::string _p = (sub); \
        std::string _d = (_s.find(_p) == std::string::npos) \
            ? std::string("substring \"") + _p + "\" not found" : ""; \
        nnc_test::checkImpl(_s.find(_p) != std::string::npos, \
            "contains(\"" sub "\")", _d, __FILE__, __LINE__); \
    } while(0)

#define CHECK_NOT_CONTAINS(str_expr, sub) \
    do { \
        std::string _s = (str_expr); \
        std::string _p = (sub); \
        std::string _d = (_s.find(_p) != std::string::npos) \
            ? std::string("unexpected substring \"") + _p + "\" found" : ""; \
        nnc_test::checkImpl(_s.find(_p) == std::string::npos, \
            "not_contains(\"" sub "\")", _d, __FILE__, __LINE__); \
    } while(0)

#define CHECK_THROWS(expr) \
    do { \
        bool _threw = false; \
        try { (expr); } catch (...) { _threw = true; } \
        nnc_test::checkImpl(_threw, "THROWS(" #expr ")", \
            _threw ? "" : "expected exception, none thrown", __FILE__, __LINE__); \
    } while(0)

#define CHECK_NO_THROW(expr) \
    do { \
        bool _threw = false; std::string _msg; \
        try { (expr); } catch (const std::exception& e) { _threw=true; _msg=e.what(); } \
                         catch (...) { _threw=true; _msg="(unknown)"; } \
        nnc_test::checkImpl(!_threw, "NO_THROW(" #expr ")", \
            _threw ? "threw: " + _msg : "", __FILE__, __LINE__); \
    } while(0)

// Объявление теста: TEST(SuiteName, TestName) { ... }
#define TEST(suite, name) \
    static void _test_##suite##_##name(); \
    static nnc_test::Registrar _reg_##suite##_##name(#suite, #name, _test_##suite##_##name); \
    static void _test_##suite##_##name()

#define RUN_ALL_TESTS() nnc_test::runAll()
