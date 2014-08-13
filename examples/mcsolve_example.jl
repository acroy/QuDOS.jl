# Quantum dynamics of a damped harmonic oscillator using the mcsolve.jl.
using QuDOS
import Winston: plot, oplot, xlabel, ylabel

# parameters
n = 20
gamma = 0.1
nsteps = 50
dt = 0.25

ntraj = 10	# number of trajectories

# initialize operators etc
ad = QuDOS.creationop( n )
h = ad*ad'

x = QuDOS.positionop( n )
p = QuDOS.momentumop( n )

cs = QuDOS.coherentstatevec(n, 1.)
rho = QuDOS.QuState(cs)

# construct Lindblad QME from Hamiltonian h and
# Lindblad operator(s)
qme = QuDOS.LindbladQME( h, {sqrt(gamma)*ad',})

# propagator for QME
# lb_prop = QuDOS.QuFixedStepPropagator( qme, dt)
# or
# lb_prop = QuDOS.QuKrylovPropagator( qme )

# propagation loop
xex = zeros(nsteps+1)
pex = zeros(nsteps+1)

xex[1] = real(QuDOS.expectationval(rho, x))
pex[1] = real(QuDOS.expectationval(rho, p))

# for step=1:nsteps
# 	rho = QuDOS.propagate(lb_prop, rho, tspan=[0., dt])
#
# 	xex[step+1] = real(QuDOS.expectationval(rho, x))
# 	pex[step+1] = real(QuDOS.expectationval(rho, p))
# end
tlist=[1:nsteps]*dt
measms = Array(AbstractMatrix{Complex128},2)
measms[1]=complex(x)
measms[2]=complex(p)
exvals = QuDOS.mcsolve(qme, cs, tlist, measms, ntraj)

xex[2:end] = real(exvals[1,1,:])
pex[2:end] = real(exvals[1,2,:])

# plots results using Winston
plot([0,tlist], xex, "b-", [0,tlist], pex, "r-")
oplot(dt*[0:0.5:nsteps], sqrt(2.)*exp(-(gamma/2.)*dt*[0:0.5:nsteps]),"k--")
oplot(dt*[0:0.5:nsteps],-sqrt(2.)*exp(-(gamma/2.)*dt*[0:0.5:nsteps]),"k--")

xlabel("time")
ylabel("\\langle x\\rangle (blue) and \\langle p\\rangle (red)")
