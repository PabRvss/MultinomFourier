// This assumes len, lprob_term, llr_term, pr, ipt, ipt0, bounds, stat_llr, newProbs_llr are
// declared by the includer.

static double newProbs_llr_examined_m = 0;    // per-shell total probability examined, either side

static inline void accumulate_llr_modified(double prob, double llr_value) {
    newProbs_llr_examined_m += prob;
    if (llr_value >= stat_llr) newProbs_llr += prob;
}

void recurse_llr_modified(int rp,int rm,int n_cat,double lprob,double llr){
    if(n_cat > 2){
        int lenN = n_cat*len;
        int c = ipt[n_cat] + rm;
        int c1 = 0;
        if(c < 0){
            c1 += c;
            c = 0;
        }
        c += lenN;
        for(int newrm = c1; newrm >= rm; newrm--){
            recurse_llr_modified(rp,newrm,n_cat-1,
                    lprob + lprob_term[c],
                    llr + llr_term[c]);
            c++;
        }
        for(int newrp = rp-1; newrp >= 0; newrp--){
            recurse_llr_modified(newrp,rm,n_cat-1,
                    lprob + lprob_term[c],
                    llr + llr_term[c]);
            c++;
        }
    }
    if(n_cat == 2){
        int len2 = 2*len;
        int c = len2 + ipt[2];
        int c1 = len + ipt[1];
        int c0 = ipt[0];

        if(rp == 0){
            if(rm == 0){
                double prob = std::exp(lprob + lprob_term[c] + lprob_term[c1] + lprob_term[c0]);

                accumulate_llr_modified(prob, llr + llr_term[c] + llr_term[c1] + llr_term[c0]);
            }
            else{
                c += rm;

                for(int newrm = 0;newrm >= rm;newrm--){
                    if(c >= len2) recurse_llr_modified(0,newrm,1,lprob + lprob_term[c],llr + llr_term[c]);
                    c++;
                }
            }
        }
        else if(rm == 0){
            for(int newrp = rp;newrp >= 0;newrp--){
                recurse_llr_modified(newrp,0,1,lprob + lprob_term[c],llr + llr_term[c]);
                c++;
            }
        }
        else{
            int c_pm = c + rm;
            int c1_pm = c1 + rp;
            int c0_pm = c0;

            int c_mp;
            int c1_mp = c1;
            int c0_mp = c0 + rp;

            if(c_pm >= len2){
                recurse_llr_modified(rp,0,1,lprob + lprob_term[c_pm],llr + llr_term[c_pm]);
                c_pm++;
                c0_pm--;
                c1_mp--;
            }
            else{
                c_pm -= len2;
                if(c_pm < 0){
                    c0_pm += c_pm;
                    c1_mp += c_pm;
                    c_pm = 0;
                }
                c_pm += len2;
            }
            c_mp = c_pm;

            //pm: Add to cat 1, subtract from cat 0
            if(c0_pm >= 0){
                double prob_pm = std::exp(lprob + lprob_term[c_pm] + lprob_term[c1_pm] + lprob_term[c0_pm]);

                while(c_pm < c && c0_pm >= 0){
                    accumulate_llr_modified(prob_pm, llr + llr_term[c_pm] + llr_term[c1_pm] + llr_term[c0_pm]);

                    c_pm++;
                    prob_pm *= ((c0_pm)/pr[0]*pr[2]/(c_pm-len2));
                    c0_pm--;
                }
                if(ipt[0] >= -rm){
                    while(c_pm < c + rp){
                        accumulate_llr_modified(prob_pm, llr + llr_term[c_pm] + llr_term[c1_pm] + llr_term[c0_pm]);

                        c_pm++;
                        prob_pm *= ((c1_pm-len)/pr[1]*pr[2]/(c_pm-len2));
                        c1_pm--;
                    }
                }
            }

            //mp: Subtract from cat 1, add to cat 0
            if(c1_mp >= len){
                double prob_mp = std::exp(lprob + lprob_term[c_mp] + lprob_term[c1_mp] + lprob_term[c0_mp]);

                while(c_mp < c && c1_mp >= len){
                    accumulate_llr_modified(prob_mp, llr + llr_term[c_mp] + llr_term[c1_mp] + llr_term[c0_mp]);

                    c_mp++;
                    prob_mp *= ((c1_mp-len)/pr[1]*pr[2]/(c_mp-len2));
                    c1_mp--;
                }
                if(ipt[1] >= -rm){
                    while(c_mp < c + rp){

                        accumulate_llr_modified(prob_mp, llr + llr_term[c_mp] + llr_term[c1_mp] + llr_term[c0_mp]);

                        c_mp++;
                        prob_mp *= ((c0_mp)/pr[0]*pr[2]/(c_mp-len2));
                        c0_mp--;
                    }
                }
            }

            if(ipt[1]+ipt[0] >= -rm) recurse_llr_modified(0,rm,1,lprob + lprob_term[c+rp],llr + llr_term[c+rp]);

        }

    }
    if(n_cat == 1){
        if(rp == 0){
            int c1 = ipt[1] + len;
            int c0 = ipt[0] + rm;
            if(c0 < 0){
                c1 += c0;
                c0 = 0;
            }

            double prob = std::exp(lprob + lprob_term[c1] + lprob_term[c0]);

            while(c1 >= len && c0 <= ipt[0]){
                accumulate_llr_modified(prob, llr + llr_term[c1] + llr_term[c0]);

                c0++;
                prob *= ((c1-len)/pr[1]*pr[0]/c0);
                c1--;
            }
        }
        else if(rm == 0){
            int c1 = ipt[1] + len;
            int c0 = ipt[0] + rp;

            double prob = std::exp(lprob + lprob_term[c1] + lprob_term[c0]);

            while(c0 >= ipt[0]){
                accumulate_llr_modified(prob, llr + llr_term[c1] + llr_term[c0]);

                c1++;
                prob *= ((c0)/pr[0]*pr[1]/(c1-len));
                c0--;
            }
        }
    }
}

// Adjusted recursive function which allows to avoid enumerating
// areas with negligible probability, only replaces the part
// of recurse for n_cat > 2. Do not call with n_cat < 3!
void recurse0_llr_modified(int rp,int rm,int n_cat,double lprob,double llr){
    int lenN = n_cat*len;
    int c = lenN;

    for(int newrp = rp; newrp >= 0; newrp--){
        if(bounds[c]){
            if(ipt0[n_cat-1] && n_cat > 3){
                recurse0_llr_modified(newrp,rm,n_cat-1,
                         lprob + lprob_term[c],
                         llr + llr_term[c]);
            }
            else recurse_llr_modified(newrp,rm,n_cat-1,
                         lprob + lprob_term[c],
                         llr + llr_term[c]);
        }
        c++;
    }
}

//' @title C++ Function for Exact Multinomial Test (numerically-stabilized LLR p-value, subtractive remaining-mass stopping criterion)
//'
//' @description Same algorithm as multinom_llr_stable.cpp's multinom_test_llr_stable, with one
//' further change: the loop's stopping criterion tracks remaining (unexamined) probability mass
//' via subtraction from 1.0, instead of comparing an accumulated examined-mass sum to 1-theta --
//' avoiding the floating-point saturation that can silently cap the achievable precision of the
//' stable variant regardless of how small theta is requested. Does not perform any safety checks.
//'
//' @param x Vector of non-negative integers - number of times each outcome was observed.
//' @param p A vector of positive numbers - the hypothesized probabilities for each outcome. Need to sum to 1!
//' @param theta Parameter - the loop stops once no more than theta worth of probability mass
//' remains unclassified; the returned value is an upper bound (like the original's early
//' termination behavior) whose gap from the true p-value is at most that remaining mass -- unless
//' out_reliable comes back false, see below.
//' @param out_reliable Optional (may be null). Set to false when remaining_mass went negative by
//' theta or more before being clamped to zero for the return value, which happens when
//' accumulated floating-point error (from summing many composition probabilities across many
//' shells) exceeds the true remaining probability mass before theta's target is genuinely
//' reached -- i.e. theta was set below this instance's own numerical noise floor. In that case the
//' returned value is just whatever tail mass had been explicitly classified up to that point, not
//' a bound with the requested theta guarantee.
//'
//' @details The outcomes should be ordered by the hypothesized probabilities from largest to smallest for optimal performance.
//'
//' @return Returns the LLR-statistic p-value.
double multinom_test_llr_modified(IntegerVector& x, NumericVector& p, double theta = 0.0001, bool* out_reliable = nullptr){ // assumes x and p to be ordered in decreasing order of p
    int n = x.sum();
    int m = p.size();
    len = n+1;

    // precompute values which are used frequently
    double *lfac = new double[n+1];
    lfac[0] = 0;
    for(int i = 0;i<n;i++){
        lfac[i+1] = lfac[i] + std::log(i+1);
    }
    double *lint = new double[n+1];
    lint[0] = 0;
    for(int i = 1;i<n+1;i++){
        lint[i] = std::log(i);
    }
    double *lp = new double[m];
    for(int j = 0;j<m;j++){
        lp[j] = std::log(p[j]);
    }
    double *expect = new double[m];
    for(int j = 0;j<m;j++){
        expect[j] = p[j]*n;
    }
    double *expect_inv = new double[m];
    for(int j = 0;j<m;j++){
        expect_inv[j] = 1/expect[j];
    }
    double *lexpect = new double[m];
    for(int j = 0;j<m;j++){
        lexpect[j] = std::log(expect[j]);
    }

    // precompute all possible summands of the test statistics
    lprob_term = new double[m*len];
    for(int j = 0;j<m;j++){
        for(int i = 0;i<n+1;i++){
            lprob_term[len*j + i] = i*lp[j] - lfac[i];
        }
    }
    llr_term = new double[m*len];
    for(int j = 0;j<m;j++){
        for(int i = 0;i<n+1;i++){
            llr_term[len*j + i] = 2*i*(lint[i] - lexpect[j]);
        }
    }

    // compute Chisq and LLR values for observation
    double obs_llr = 0;
    for(int j = 0;j<m;j++){
        obs_llr += llr_term[len*j + x[j]];
    }

    // compute center (point with integer values close to the expectation)
    NumericVector expected = n*p;
    NumericVector init_point = NumericVector(expected);
    for (int i=0; i<m; i++) init_point[i] = std::floor(init_point[i]);
    NumericVector decimalplaces = expected - init_point;
    double missing = n - init_point.sum();
    for(int i = 0;i < missing;i++){
        init_point[i]++;
    }
    double init_lprob = lfac[n];
    double obs_lprob = lfac[n];
    for(int i = 0;i < m;i++){
        init_lprob += init_point[i]*std::log(p[i]) - lfac[int(init_point[i])];
        obs_lprob += x[i]*std::log(p[i]) - lfac[x[i]];
    }
    if(obs_lprob >= init_lprob){
        std::copy(begin(x), end(x), begin(init_point));
        init_lprob = obs_lprob;
    }

    // write p and initial point to global variables accessible by the recursive function
    pr = new double[m];
    ipt = new int[m];
    for(int j = 0;j<m;j++){
        lp[j] = std::log(p[j]);
        pr[j] = p[j];
        ipt[j] = init_point[j];
    }

    // set boundary for the LLR test statistic
    stat_llr = obs_llr - 0.00000001*fabs(obs_llr);

    // bounds for small expectations to avoid enumerating many points with negligible probability
    ipt0 = new bool[m];
    bounds = new bool[m*len];
    for(int j = 0;j<m;j++){
        ipt0[j] = (ipt[j] == 0);
        if(ipt0[j]){
            double pbinom = 0; // null probability to observe at least i in category j
            double lp_comp = std::log(1-pr[j]); // complementary probability
            for(int i = n;i > -1;i--){
                pbinom += std::exp(lfac[n] - lfac[i] - lfac[n-i] + i*lp[j] + (n-i)*lp_comp);
                bounds[len*j + i] = pbinom > theta * 0.00000001;
            }
        }
    }

    // initial values
    double sumProbs_llr = 0;

    // remaining (unexamined) probability mass: starts at 1.0, decremented each shell. Comparing
    // this directly to theta avoids the saturation issue.
    double remaining_mass = 1.0;

    double lprob;
    double llr;

    int r = 0;

    do{
        newProbs_llr = 0;
        newProbs_llr_examined_m = 0;

        lprob = lfac[n];
        llr = 0;

        if(m == 2){ // binomial case
            int c1 = ipt[1] + len - r;
            int c0 = ipt[0] + r;
            if(c0 < len){
                double prob = std::exp(lprob + lprob_term[c1] + lprob_term[c0]);
                accumulate_llr_modified(prob, llr + llr_term[c1] + llr_term[c0]);
            }
            c1 += 2*r;
            c0 -= 2*r;
            if(c0 >= 0 && r > 0){
                double prob = std::exp(lprob + lprob_term[c1] + lprob_term[c0]);
                accumulate_llr_modified(prob, llr + llr_term[c1] + llr_term[c0]);
            }
        }
        else if(ipt0[m-1] && m > 3) recurse0_llr_modified(r,-r,m-1,lprob,llr);
        else recurse_llr_modified(r,-r,m-1,lprob,llr);

        sumProbs_llr += newProbs_llr;
        remaining_mass -= newProbs_llr_examined_m;

        r++;
    } while ((newProbs_llr_examined_m > 0 || r < 2)
             && remaining_mass > theta && r < n);

    delete[] lprob_term;
    delete[] llr_term;
    delete[] ipt;
    delete[] pr;
    delete[] ipt0;
    delete[] bounds;
    delete[] lfac;
    delete[] lint;
    delete[] lp;
    delete[] expect;
    delete[] expect_inv;
    delete[] lexpect;

    if (out_reliable) *out_reliable = (remaining_mass >= -theta);
    const double remaining_nonneg = remaining_mass > 0 ? remaining_mass : 0;
    return sumProbs_llr + remaining_nonneg;
}
