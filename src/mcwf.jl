
# quantum state as an ensemble of pure states
type QuStateEnsemble{S<:AbstractQuState}
  state::S
  decomp

  QuStateEnsemble(rho::S, d) = new(rho, d)
end

function QuStateEnsemble{S<:AbstractQuState}(rho::S, d)
  return QuStateEnsemble{S}(rho, d)
end

function QuStateEnsemble(rho::QuState)
  println("QuStateEnsemble{QuState}")
  QuStateEnsemble(rho, eigfact(reshape(full(rho.elem), rho.nb, rho.nb)))
end

function QuStateEnsemble(psi::QuStateVec)
  println("QuStateEnsemble{QuStateVec}")
  QuStateEnsemble(psi, nothing)
end

# for a mixed state we use the spectral decomposition
function draw{C<:AbstractArray{Complex128,1}}(e::QuStateEnsemble{QuState{C}})
  println("draw from QuStateEnsemble{QuState}")

  r = rand() # draw random number from [0,1)
  pacc = 0.
  for i=1:length(e.decomp[:values])
    pacc = pacc + e.decomp[:values][i]
    if pacc >= r
      return QuStateVec(complex(e.decomp[:vectors][:,i]), e.state.subnb)
    end
  end
end

# for a pure state drawing is easy
function draw{C<:AbstractArray{Complex128,1}}(e::QuStateEnsemble{QuStateVec{C}})
  println("draw from QuStateEnsemble{QuStateVec}")

  return copy(e.state)
end


# MCWF method
#
# inputs: qme::AbstractLindbladQME
#         state::AbstractQuState - initial state
#         ntraj::Int    - number of trajectories
#         dt0::Float64  - time-step
#         nsteps::Int   - number of time-steps
#         ops           - Array of operators for which
#                         expectation values are calcluated at each time-step
# outputs: rho::QuState - final density matrix after nsteps
#          exvals       - nsteps*nops matrix with expectation values
#                         for each time-step and each op in ops
#
function mcwfpropagate(state::AbstractQuState,
                        qme::AbstractLindbladQME,
                        ntraj::Int, dt0::Float64, nsteps::Int,
                        ops=[])

    jtol = 1.e-6  # jump tolerance

    nops = length(ops)
    if nops > 0
        exvals = Array(Complex128, nsteps, nops)
        fill!(exvals, 0. + im*0.)
    else
        exvals = []
    end

    # get information of QME
    heff = eff_hamiltonian(qme)
    c = linbladops(qme)

    # final state
    rho = QuState( zeros(Complex128, state.nb, state.nb), state.subnb )

    # initial state -> ensemble
    qse = QuStateEnsemble(state)

    for traj=1:ntraj
	    println("-- starting trajectory $traj/$ntraj")
        t = 0.

        # initial state vec for trajectory
	    psi  = draw(qse)

        eps = rand() # draw random number from [0,1)

	    for step=1:nsteps

		    dt = dt0
		    accdt = 0.
            tj = 0.

		    while accdt < dt0

			    # propagate one time-step
			    eff_prop = QuFixedStepPropagator( -im*heff, dt)
			    psi1 = propagate(eff_prop, psi)

                if norm(psi1)^2 > eps + jtol/2
    			    psi = copy(psi1)  # no jump
    			    accdt = accdt + dt
    			    dt = dt0 - accdt    # try to propagate for the remainder of the
                                        # time step

    		    elseif norm(psi1)^2 < eps - jtol/2
    			    dt = dt/2		# jump in the current interval

                else
    			    println("jump at t=$(tj + accdt + dt), norm^2=$(norm(psi1)^2), eps=$eps")

    			    PnS = 0.
    			    for cind = 1:length(c)
    				    PnS += real(expectationval(psi1, c[cind]'*c[cind]))
    		        end
                    println("jump at t=$(tj + accdt + dt), 1-norm=$(1-norm(psi1)), PnS=$PnS")

                    Pn = 0.
    			    cop = speye(size(heff,1))
    			    for cind = 1:length(c)
    				    Pn += real(expectationval(psi1, c[cind]'*c[cind]))/PnS
    				    println("Pn($cind)=$Pn")

    				    if Pn >= eps
    					    println("using operator $cind")
    					    cop=c[cind]
    					    break
    				    end
    			    end

                    # now we have a jump
                    psi = applyop(psi1, cop)
    			    normalize!(psi)

    			    eps = rand() # draw new random number from [0,1)
    			    accdt = accdt + dt
    			    dt = dt0 - accdt
    		    end

    		    println("dt = $(accdt+dt), norm = $(norm(psi1)), $(norm(psi))")
    	    end

            for nop = 1:nops
                exvals[step, nop] = exvals[step, nop] + expectationval(normalize(psi), ops[nop])/ntraj
            end

    	    tj = tj + dt0

            t = t + accdt
            println("== t = $(t), norm = $(norm(psi))")

        end # time-steps

        normalize!(psi)
        rho = rho + QuState(psi)/ntraj

      end  # trajectories

      return rho, exvals
end
