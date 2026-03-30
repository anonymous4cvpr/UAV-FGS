#pragma once

// #define GLM_CXX98_EXCEPTIONS
// #define GLM_CXX98_RTTI

// #define GLM_CXX11_RVALUE_REFERENCES
// Rvalue references - GCC 4.3
// XXXX

// GLM_CXX11_TRAILING_RETURN
// Rvalue references for *this - GCC not supported
// XXXX

// GLM_CXX11_NONSTATIC_MEMBER_INIT
// Initialization of class objects by rvalues - GCC any
// XXXX

// GLM_CXX11_NONSTATIC_MEMBER_INIT
// Non-static data member initializers - GCC 4.7
// XXXX

// #define GLM_CXX11_VARIADIC_TEMPLATE
// Variadic templates - GCC 4.3
// XXXX

//
// Extending variadic template template parameters - GCC 4.4
// XXXX

// #define GLM_CXX11_GENERALIZED_INITIALIZERS
// Initializer lists - GCC 4.4
// XXXX

// #define GLM_CXX11_STATIC_ASSERT
// Static assertions - GCC 4.3
// XXXX

// #define GLM_CXX11_AUTO_TYPE
// auto-typed variables - GCC 4.4
// XXXX

// #define GLM_CXX11_AUTO_TYPE
// Multi-declarator auto - GCC 4.4
// XXXX

// #define GLM_CXX11_AUTO_TYPE
// Removal of auto as a storage-class specifier - GCC 4.4
// XXXX

// #define GLM_CXX11_AUTO_TYPE
// New function declarator syntax - GCC 4.4
// XXXX

// #define GLM_CXX11_LAMBDAS
// New wording for C++0x lambdas - GCC 4.5
// XXXX

// #define GLM_CXX11_DECLTYPE
// Declared type of an expression - GCC 4.3
// XXXX

//
// Right angle brackets - GCC 4.3
// XXXX

//
// Default template arguments for function templates	DR226	GCC 4.3
// XXXX

//
// Solving the SFINAE problem for expressions	DR339	GCC 4.4
// XXXX

// #define GLM_CXX11_ALIAS_TEMPLATE
// Template aliases	N2258	GCC 4.7
// XXXX

//
// Extern templates	N1987	Yes
// XXXX

// #define GLM_CXX11_NULLPTR
// Null pointer constant	N2431	GCC 4.6
// XXXX

// #define GLM_CXX11_STRONG_ENUMS
// Strongly-typed enums	N2347	GCC 4.4
// XXXX

//
// Forward declarations for enums	N2764	GCC 4.6
// XXXX

//
// Generalized attributes	N2761	GCC 4.8
// XXXX

//
// Generalized constant expressions	N2235	GCC 4.6
// XXXX

//
// Alignment support	N2341	GCC 4.8
// XXXX

// #define GLM_CXX11_DELEGATING_CONSTRUCTORS
// Delegating constructors	N1986	GCC 4.7
// XXXX

//
// Inheriting constructors	N2540	GCC 4.8
// XXXX

// #define GLM_CXX11_EXPLICIT_CONVERSIONS
// Explicit conversion operators	N2437	GCC 4.5
// XXXX

//
// New character types	N2249	GCC 4.4
// XXXX

//
// Unicode string literals	N2442	GCC 4.5
// XXXX

//
// Raw string literals	N2442	GCC 4.5
// XXXX

//
// Universal character name literals	N2170	GCC 4.5
// XXXX

// #define GLM_CXX11_USER_LITERALS
// User-defined literals		N2765	GCC 4.7
// XXXX

//
// Standard Layout Types	N2342	GCC 4.5
// XXXX

// #define GLM_CXX11_DEFAULTED_FUNCTIONS
// #define GLM_CXX11_DELETED_FUNCTIONS
// Defaulted and deleted functions	N2346	GCC 4.4
// XXXX

//
// Extended friend declarations	N1791	GCC 4.7
// XXXX

//
// Extending sizeof	N2253	GCC 4.4
// XXXX

// #define GLM_CXX11_INLINE_NAMESPACES
// Inline namespaces	N2535	GCC 4.4
// XXXX

// #define GLM_CXX11_UNRESTRICTED_UNIONS
// Unrestricted unions	N2544	GCC 4.6
// XXXX

// #define GLM_CXX11_LOCAL_TYPE_TEMPLATE_ARGS
// Local and unnamed types as template arguments	N2657	GCC 4.5
// XXXX

// #define GLM_CXX11_RANGE_FOR
// Range-based for	N2930	GCC 4.6
// XXXX

// #define GLM_CXX11_OVERRIDE_CONTROL
// Explicit virtual overrides	N2928 N3206 N3272	GCC 4.7
// XXXX
// XXXX
// XXXX

//
// Minimal support for garbage collection and reachability-based leak detection	N2670	No
// XXXX

// #define GLM_CXX11_NOEXCEPT
// Allowing move constructors to throw [noexcept]	N3050	GCC 4.6 (core language only)
// XXXX

//
// Defining move special member functions	N3053	GCC 4.6
// XXXX

//
// Sequence points	N2239	Yes
// XXXX

//
// Atomic operations	N2427	GCC 4.4
// XXXX

//
// Strong Compare and Exchange	N2748	GCC 4.5
// XXXX

//
// Bidirectional Fences	N2752	GCC 4.8
// XXXX

//
// Memory model	N2429	GCC 4.8
// XXXX

//
// Data-dependency ordering: atomics and memory model	N2664	GCC 4.4
// XXXX

//
// Propagating exceptions	N2179	GCC 4.4
// XXXX

//
// Abandoning a process and at_quick_exit	N2440	GCC 4.8
// XXXX

//
// Allow atomics use in signal handlers	N2547	Yes
// XXXX

//
// Thread-local storage	N2659	GCC 4.8
// XXXX

//
// Dynamic initialization and destruction with concurrency	N2660	GCC 4.3
// XXXX

//
// __func__ predefined identifier	N2340	GCC 4.3
// XXXX

//
// C99 preprocessor	N1653	GCC 4.3
// XXXX

//
// long long	N1811	GCC 4.3
// XXXX

//
// Extended integral types	N1988	Yes
// XXXX

#if(GLM_COMPILER & GLM_COMPILER_GCC)

#	define GLM_CXX11_STATIC_ASSERT

#elif(GLM_COMPILER & GLM_COMPILER_CLANG)
#	if(__has_feature(cxx_exceptions))
#		define GLM_CXX98_EXCEPTIONS
#	endif

#	if(__has_feature(cxx_rtti))
#		define GLM_CXX98_RTTI
#	endif

#	if(__has_feature(cxx_access_control_sfinae))
#		define GLM_CXX11_ACCESS_CONTROL_SFINAE
#	endif

#	if(__has_feature(cxx_alias_templates))
#		define GLM_CXX11_ALIAS_TEMPLATE
#	endif

#	if(__has_feature(cxx_alignas))
#		define GLM_CXX11_ALIGNAS
#	endif

#	if(__has_feature(cxx_attributes))
#		define GLM_CXX11_ATTRIBUTES
#	endif

#	if(__has_feature(cxx_constexpr))
#		define GLM_CXX11_CONSTEXPR
#	endif

#	if(__has_feature(cxx_decltype))
#		define GLM_CXX11_DECLTYPE
#	endif

#	if(__has_feature(cxx_default_function_template_args))
#		define GLM_CXX11_DEFAULT_FUNCTION_TEMPLATE_ARGS
#	endif

#	if(__has_feature(cxx_defaulted_functions))
#		define GLM_CXX11_DEFAULTED_FUNCTIONS
#	endif

#	if(__has_feature(cxx_delegating_constructors))
#		define GLM_CXX11_DELEGATING_CONSTRUCTORS
#	endif

#	if(__has_feature(cxx_deleted_functions))
#		define GLM_CXX11_DELETED_FUNCTIONS
#	endif

#	if(__has_feature(cxx_explicit_conversions))
#		define GLM_CXX11_EXPLICIT_CONVERSIONS
#	endif

#	if(__has_feature(cxx_generalized_initializers))
#		define GLM_CXX11_GENERALIZED_INITIALIZERS
#	endif

#	if(__has_feature(cxx_implicit_moves))
#		define GLM_CXX11_IMPLICIT_MOVES
#	endif

#	if(__has_feature(cxx_inheriting_constructors))
#		define GLM_CXX11_INHERITING_CONSTRUCTORS
#	endif

#	if(__has_feature(cxx_inline_namespaces))
#		define GLM_CXX11_INLINE_NAMESPACES
#	endif

#	if(__has_feature(cxx_lambdas))
#		define GLM_CXX11_LAMBDAS
#	endif

#	if(__has_feature(cxx_local_type_template_args))
#		define GLM_CXX11_LOCAL_TYPE_TEMPLATE_ARGS
#	endif

#	if(__has_feature(cxx_noexcept))
#		define GLM_CXX11_NOEXCEPT
#	endif

#	if(__has_feature(cxx_nonstatic_member_init))
#		define GLM_CXX11_NONSTATIC_MEMBER_INIT
#	endif

#	if(__has_feature(cxx_nullptr))
#		define GLM_CXX11_NULLPTR
#	endif

#	if(__has_feature(cxx_override_control))
#		define GLM_CXX11_OVERRIDE_CONTROL
#	endif

#	if(__has_feature(cxx_reference_qualified_functions))
#		define GLM_CXX11_REFERENCE_QUALIFIED_FUNCTIONS
#	endif

#	if(__has_feature(cxx_range_for))
#		define GLM_CXX11_RANGE_FOR
#	endif

#	if(__has_feature(cxx_raw_string_literals))
#		define GLM_CXX11_RAW_STRING_LITERALS
#	endif

#	if(__has_feature(cxx_rvalue_references))
#		define GLM_CXX11_RVALUE_REFERENCES
#	endif

#	if(__has_feature(cxx_static_assert))
#		define GLM_CXX11_STATIC_ASSERT
#	endif

#	if(__has_feature(cxx_auto_type))
#		define GLM_CXX11_AUTO_TYPE
#	endif

#	if(__has_feature(cxx_strong_enums))
#		define GLM_CXX11_STRONG_ENUMS
#	endif

#	if(__has_feature(cxx_trailing_return))
#		define GLM_CXX11_TRAILING_RETURN
#	endif

#	if(__has_feature(cxx_unicode_literals))
#		define GLM_CXX11_UNICODE_LITERALS
#	endif

#	if(__has_feature(cxx_unrestricted_unions))
#		define GLM_CXX11_UNRESTRICTED_UNIONS
#	endif

#	if(__has_feature(cxx_user_literals))
#		define GLM_CXX11_USER_LITERALS
#	endif

#	if(__has_feature(cxx_variadic_templates))
#		define GLM_CXX11_VARIADIC_TEMPLATES
#	endif

#endif//(GLM_COMPILER & GLM_COMPILER_CLANG)
