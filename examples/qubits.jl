# example involving a two qubit state, negativity, fidelity etc.

using QuDOS
import Winston: plot, oplot, xlabel, ylabel

# construct a pure 2 qubit state from coefficents such that
#   |psi> = ( a |00> + b |01> + c |10> + d |11>)/sqrt(norm)
#
function puretwoqubit(a::Number, b::Number, c::Number, d::Number)

  # ground state single qubit
  fs0 = fockstatevec(2,1)

  # excited state single qubit
  fs1 = fockstatevec(2,2)

  fs00 = tensor(fs0, 2)    # equiv. to tensor(fs0, fs0)
  fs01 = tensor(fs0, fs1)
  fs10 = tensor(fs1, fs0)
  fs11 = tensor(fs1, 2)

  return normalize!( a*fs00 + b*fs01 + c*fs10 + d*fs11)
end

###############################################################################
# construct a Bell state
bell = QuState( puretwoqubit(0., 1., 1., 0.) )

println("purity : ", purity(bell))
# negativity should be N(rho) = |a*d-b*c|, which is 1/2 for the bell state
println("negativity : ", negativity(bell, 1))

###############################################################################
gamma = 0.1
nsteps = 50
dt = 0.25


# denoting the two subsystems by A and B
# we can construct a dephasing QME
nop = sparse([2,], [2,], [1.,])
nopA = kron(nop, speye(2))
nopB = kron(nop, speye(2))

qme = LindbladQME(spzeros(4,4), {sqrt(gamma)*nopA, sqrt(gamma)*nopB,})
# propagator for QME
lb_prop = QuDOS.QuFixedStepPropagator( qme, dt)
# or
# lb_prop = QuDOS.QuKrylovPropagator( qme )

neg = zeros(nsteps+1)
neg[1] = negativity(bell, 1)

# propagation loop
rho = bell
for step=1:nsteps
  rho = QuDOS.propagate(lb_prop, rho, tspan=[0., dt])

  neg[step+1] = negativity(rho, 1)
end

println("fidelity with Bell state t=0: ", fidelity(bell, bell))
println("fidelity with Bell state t=$(nsteps*dt): ", fidelity(bell, rho))

# plots results using Winston
plot(dt*[0:nsteps], neg, "b-")
oplot(dt*[0:5:nsteps], 0.5*exp(-gamma*dt*[0:5:nsteps]), "rs")

xlabel("time")
ylabel("negativity")
