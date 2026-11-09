/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

typedef struct {    // Internals are *Private*.
    ulong state;    // RNG state.  All values are possible.
    ulong inc;      // Controls which RNG sequence (stream) is selected. Must *always* be odd.
} pcg32_random_t;

#ifdef USE_DOUBLE_PRECISION
#define FACTOR32 0x1p-32
#else
#define FACTOR32 0x1.fffffep-33
//#define FACTOR32 0x1p-24
#endif
// pcg32_random_r(rng)
//     Generate a uniformly distributed 32-bit random number
static inline uint pcg32_random_r(pcg32_random_t* rng) {
    ulong oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005UL + rng->inc;
    uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
//    return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) >> 8;
}

// pcg32_srandom_r(rng, initstate, initseq):
//     Seed the rng.  Specified in two parts, state initializer and a
//     sequence selection constant (a.k.a. stream id)
static inline void pcg32_srandom_r(pcg32_random_t* rng, ulong initstate, ulong initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}
