#pragma once

// adapted from https://www.stat.uchicago.edu/~mcpeek/software/CCQLSpackage1.3/

#define        Z_MAX          6.0            /* maximum meaningful z value */
#define        LOG_SQRT_PI    0.5723649429247000870717135 /* log(sqrt(pi)) */
#define        I_SQRT_PI      0.5641895835477562869480795 /* 1/sqrt(pi) */
#define        BIGX           20.0         /* max value to represent exp (x) */

static double normal(double z) {
    double  y, x, w;

    if (z == 0.0) x = 0.0;
    else {
        y = 0.5*fabs(z);
        if (y >= (Z_MAX*0.5)) x = 1.0;
        else if (y < 1.0) {
            w = y*y;
            x = ((((((((0.000124818987 * w
                        -0.001075204047) * w +0.005198775019) * w
                      -0.019198292004) * w +0.059054035642) * w
                    -0.151968751364) * w +0.319152932694) * w
                  -0.531923007300) * w +0.797884560593) * y * 2.0;
        }
        else {
            y -= 2.0;
            x = (((((((((((((-0.000045255659 * y
                             +0.000152529290) * y -0.000019538132) * y
                           -0.000676904986) * y +0.001390604284) * y
                         -0.000794620820) * y -0.002034254874) * y
                       +0.006549791214) * y -0.010557625006) * y
                     +0.011630447319) * y -0.009279453341) * y
                   +0.005353579108) * y -0.002141268741) * y
                 +0.000535310849) * y +0.999936657524;
        }
    }
    return z > 0.0 ? (x + 1.0) * 0.5 : (1.0 - x) * 0.5;
}

static inline double ex(double x) {return (x < -BIGX) ? 0.0 : exp(x);}

static double pochisq(double x, small_t df) {
    double a, y=0.0, s;
    double e, c, z;
    bool even;

    if (x <= 0.0 || df < 1) return 1.0;

    a = 0.5 * x;
    even = (df + 1) % 2;
    if (df > 1) y = ex(-a);
    s = even ? y : 2.0 * normal(-sqrt (x));
    if (df > 2) {
        x = 0.5 * (df - 1.0);
        z = even ? 1.0 : 0.5;
        if (a > BIGX) {
            e = even ? 0.0 : LOG_SQRT_PI;
            c = log (a);
            while (z <= x) {
                e = log (z) + e;
                s += ex(c*z-a-e);
                z += 1.0;
            }
            return s;
        }
        else {
            e = even ? 1.0 : (I_SQRT_PI / sqrt (a));
            c = 0.0;
            while (z <= x) {
                e = e * (a / z);
                c = c + e;
                z += 1.0;
            }
            return c * y + s;
        }
    }
    else return s;
}

static double inverse_normal(double  y) {
    double  r, s;
    if(y < 0.02425) {
        s = sqrt(-2.0 * log(y));
        return  + (2.938163982698783  +
                   (4.374664141464968  -
                    (2.549732539343734  +
                     (2.400758277161838  +
                      (0.3223964580411365 +
                       0.007784894002430293 * s) * s) * s) * s) * s)
                / (1.0                +
                   (3.754408661907416  +
                    (2.445134137142996  +
                     (0.3224671290700398 +
                      0.007784695709041462 * s) * s) * s) * s);
    }
    else if(y > 0.97575) {
        s = sqrt(-2.0 * log(1.0 - y));
        return  - (2.938163982698783  +
                   (4.374664141464968  -
                    (2.549732539343734  +
                     (2.400758277161838  +
                      (0.3223964580411365 +
                       0.007784894002430293 * s) * s) * s) * s) * s)
                / (1.0                +
                   (3.754408661907416  +
                    (2.445134137142996  +
                     (0.3224671290700398 +
                      0.007784695709041462 * s) * s) * s) * s);
    }
    else {
        s = y - 0.5;
        r = s * s;
        return  s * (2.506628277459239 -
                     (30.66479806614716 -
                      (138.3577518672690 -
                       (275.9285104469687 -
                        (220.9460984245205 -
                         39.69683028665376 * r) * r) * r) * r) * r)
                / (1.0 -
                   (13.28068155288572 -
                    (66.80131188771972 -
                     (155.6989798598866 -
                      (161.5858368580409 -
                       54.47609879822406 * r) * r) * r) * r) * r);
    }
}

static bool wilson_score(double p, size n) {
    double z = z_confidence;
    double factor = 1.0/(1.0 + z*z/n);
    double w = z*factor*sqrt(p*(1-p)/n + z*z/(4.0*n*n));
    double p0 = factor*(p + z*z/(2.0*n));

    double res[2];
    res[0] = p0 - w; if (res[0] < 0.0) res[0] = 0.0;
    res[1] = p0 + w;
    double rel = res[1]/res[0];
    if (verbose) printf("\t\tlog10(p) in [%.3f, %.3f], log10(rel) = %.3f\n", log10(res[0]), log10(res[1]), log10(rel));
    return (rel <= relative_error);
}

#define MIN(a, b) (((a)<(b))?(a):(b))
static small_t random_binomial_btpe(small_t n, double p, pcg32_random_t* rng) {
    double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
    double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
    small_t m, y, k, i;
    
    /* initialize */
    r = MIN(p, 1.0 - p);
    q = 1.0 - r;
    fm = n * r + r;
    m = floor(fm);
    p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5;
    xm = m + 0.5;
    xl = xm - p1;
    xr = xm + p1;
    c = 0.134 + 20.5 / (15.3 + m);
    a = (fm - xl) / (fm - xl * r);
    laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    lamr = a * (1.0 + a / 2.0);
    p2 = p1 * (1.0 + 2.0 * c);
    p3 = p2 + c / laml;
    p4 = p3 + c / lamr;

    /* sigh ... */
    Step10:
    nrq = n * r * q;
    u = pcg32_random_double(rng) * p4;
    v = pcg32_random_double(rng);

    if (u > p1)
        goto Step20;
    y = floor(xm - p1 * v + u);
    goto Step60;

    Step20:
    if (u > p2)
        goto Step30;
    x = xl + (u - p1) / c;
    v = v * c + 1.0 - fabs(m - x + 0.5) / p1;
    if (v > 1.0)
        goto Step10;
    y = floor(x);
    goto Step50;

    Step30:
    if (u > p3)
        goto Step40;
    y = floor(xl + log(v) / laml);
    /* Reject if v==0.0 since previous cast is undefined */
    if ((y < 0) || (v == 0.0))
        goto Step10;
    v = v * (u - p2) * laml;
    goto Step50;

    Step40:
    y = floor(xr - log(v) / lamr);
    /* Reject if v==0.0 since previous cast is undefined */
    if ((y > n) || (v == 0.0))
        goto Step10;
    v = v * (u - p3) * lamr;

    Step50:
    k = abs(y - m);
    if ((k > 20) && (k < ((nrq) / 2.0 - 1)))
        goto Step52;

    s = r / q;
    a = s * (n + 1);
    F = 1.0;
    if (m < y) {
        for (i = m + 1; i <= y; i++) {
            F *= (a / i - s);
        }
    } else if (m > y) {
        for (i = y + 1; i <= m; i++) {
            F /= (a / i - s);
        }
    }
    if (v > F)
        goto Step10;
    goto Step60;

    Step52:
    rho = (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
    t = -k * k / (2 * nrq);
    /* log(0.0) ok here */
    A = log(v);
    if (A < (t - rho))
        goto Step60;
    if (A > (t + rho))
        goto Step10;

    x1 = y + 1;
    f1 = m + 1;
    z = n + 1 - m;
    w = n - y + 1;
    x2 = x1 * x1;
    f2 = f1 * f1;
    z2 = z * z;
    w2 = w * w;
    if (A > (xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) +
             (y - m) * log(w * r / (x1 * q)) +
             (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 /
             166320. +
             (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z /
             166320. +
             (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 /
             166320. +
             (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w /
             166320.)) {
        goto Step10;
    }

    Step60:
    if (p > 0.5) {
        y = n - y;
    }

    return y;
}

static small_t random_binomial_inversion(small_t n, double p, pcg32_random_t* rng) {
    double q, qn, np, px, U;
    small_t X, bound;

    q = 1.0 - p;
    qn = exp(n * log(q));
    np = n * p;
    bound = MIN(n, np + 10.0 * sqrt(np * q + 1));
    X = 0;
    px = qn;
    U = pcg32_random_double(rng);

    while (U > px) {
        X++;
        if (X > bound) {
            X = 0;
            px = qn;
            U = pcg32_random_double(rng);
        } else {
            U -= px;
            px = ((n - X + 1) * p * px) / (X * q);
        }
    }
    return X;
}

static inline small_t random_binomial(double p, small_t n, pcg32_random_t* rng) {
    if ((n == 0LL) || (p <= 0.0f)) return 0;

    if (p <= 0.5) {
        if (p*n <= 30.0) return random_binomial_inversion(n, p, rng);
        else return random_binomial_btpe(n, p, rng);
    } else {
        double q = 1.0 - p;
        if (q*n <= 30.0) return n - random_binomial_inversion(n, q, rng);
        else return n - random_binomial_btpe(n, q, rng);
    }
}
