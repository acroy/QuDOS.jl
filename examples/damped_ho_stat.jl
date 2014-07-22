# quantum dynamics of a haromonic oscillator coupled to
# a finite temperature bath
using QuDOS
import Winston: plot, oplot, xlabel, ylabel

tic()

# parameters
n = 19
Omega = 1.
alpha = 1.
gamma = 0.1
betas = Float64[0.5, 1.0, 2., 5., 10.] # inverse temperature
nbose(om, beta)=1/(exp(beta*om)-1)


# initialize operators etc
ad = QuDOS.creationop( n )
h = Omega*ad*ad'	# Hamiltonian

x = QuDOS.positionop( n )
p = QuDOS.momentumop( n )

x2 = x*x
p2 = p*p

# storage of results
xex = Array(Float64, 0)
pex = Array(Float64, 0)

x2ex = Array(Float64, 0)
p2ex = Array(Float64, 0)

nex = Array(Float64, 0)

# main loop:
# sweep initial displacement
for beta in betas

	# start in coherent state
	cs = QuDOS.coherentstatevec(n, alpha)
	rho = QuDOS.QuState(cs)

	# construct Lindblad QME from Hamiltonian h and
	# Lindblad operator(s)
	qme = QuDOS.LindbladQME(h, {sqrt(gamma*(1+nbose(Omega,beta)))*ad',sqrt(gamma*nbose(Omega,beta))*ad})

	# find steady state
	rho = QuDOS.stationary(qme, rho, tol=1e-8)

	# collect expectation values
	push!(nex, real(QuDOS.expectationval(rho, ad*ad')))

	push!(xex, real(QuDOS.expectationval(rho, x)))
	push!(pex, real(QuDOS.expectationval(rho, p)))

	push!(x2ex, real(QuDOS.expectationval(rho, x2)))
	push!(p2ex, real(QuDOS.expectationval(rho, p2)))

end

toc()

# plots results using Winston
plot(betas, nex, "b-")
oplot(betas, Float64[nbose(Omega, beta) for beta in betas] , "bs")

xlabel("inverse temperature \\beta")
ylabel("occupation \\langle n\\rangle")
