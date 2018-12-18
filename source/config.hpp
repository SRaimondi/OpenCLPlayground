//
// Created by simon on 12/18/2018.
//

#ifndef OPENCLPLAYGROUND_CONFIG_HPP
#define OPENCLPLAYGROUND_CONFIG_HPP

// Find the version of C++
#define CXX_98  199711L
#define CXX_11  201103L
#define CXX_14  201402L
#define CXX_17  201702L

#ifndef CXX_VERSION
#if __cplusplus >= CXX_98
#define CXX_VERSION CXX_98
#endif
#if __cplusplus >= CXX_11
#undef CXX_VERSION
#define CXX_VERSION CXX_11
#endif
#if __cplusplus >= CXX_14
#undef CXX_VERSION
#define CXX_VERSION CXX_14
#endif
#if __cplusplus >= CXX_17
#undef CXX_VERSION
#define CXX_VERSION CXX_17
#endif
#endif

// Check if the CXX_VERSION was determined
#ifndef CXX_VERSION
#error "Could not define the C++ standard version"
#endif

// Check we have at least C++11
#if CXX_VERSION < CXX_11
#error "C++ version is too old, minimum required is C++11"
#endif

// Define inline keyword
#ifndef CXX_INLINE
#define CXX_INLINE inline
#endif

// Define noexcept keyword
#if CXX_VERSION >= CXX_11
#define CXX_NOEXCEPT noexcept
#else
#define CXX_NOEXCEPT
#endif

// Check if constexpr is available
#ifndef CXX_CONSTEXPR
#if CXX_VERSION >= CXX_11
#define CXX_CONSTEXPR constexpr
#else
#define CXX_CONSTEXPR CXX_INLINE
#endif
#endif

// Check if we can use constexpr for C++ 14 / 17
#ifndef CXX14_CONSTEXPR
#if CXX_VERSION >= CXX_14
#define CXX14_CONSTEXPR constexpr
#else
#define CXX14_CONSTEXPR CXX_INLINE
#endif
#endif

#ifndef CXX17_CONSTEXPR
#if CXX_VERSION >= CXX_17
#define CXX17_CONSTEXPR constexpr
#else
#define CXX17_CONSTEXPR CXX_INLINE
#endif
#endif

// Alignment specifier
#if CXX_VERSION >= CXX_11
#define CXX_ALIGNAS(alignment) alignas(alignment)
#else
#define CXX_ALIGNAS(alignment)
#endif

#endif //OPENCLPLAYGROUND_CONFIG_HPP
