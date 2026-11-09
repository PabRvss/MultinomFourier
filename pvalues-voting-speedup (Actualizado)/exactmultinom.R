library(ExactMultinom)

x = c(20, 20, 10, 10, 5, 5, 2, 2)
p = rep(1/length(x),length(x))
theta = 1e-100

print(x)
print(sum(x))

t0 <- Sys.time()
multinom.test(x,p,theta = theta, timelimit = 300)
# multinom.test(x,p,theta = theta,method = "asymptotic")
# multinom.test(x,p,theta = theta,method = "Monte-Carlo")
t1 <- Sys.time()
print(t1-t0)
