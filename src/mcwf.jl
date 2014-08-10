# MCWF method
#
# inputs: qme::AbstractLindbladQME
#         psi0::QuStateVec
#         ntraj::Int    - number of trajectories
#         dt0::Float64  - time-step
#         nsteps::Int   - number of time-steps
#         ops           - Array of operators for which
#                         expectation values are calcluated at each time-step
# outputs: rho::QuState - final density matrix after nsteps
#          exvals       - nsteps*nops matrix with expectation values
#                         for each time-step and each op in ops
#
function mcwfpropagate(psi0::QuStateVec,
                        qme::AbstractLindbladQME,
                        ntraj::Int, dt0::Float64, nsteps::Int,
                        ops=[])

  jtol = 1.e-6  # jump tolerance

  nops = length(ops)
  if nops > 0
    exvals = Array(Complex128, nsteps, nops)
  else
    exvals = []
  end

  # get information of QME
  heff = eff_hamiltonian(qme)
  c = linbladops(qme)

  rho = QuState( zeros(Complex128, psi0.nb, psi0.nb), psi0.subnb )

  for traj=1:ntraj
	  println("starting trajectory $traj/$ntraj")

	  eps = rand() # draw random number from [0,1)
	  psi  = copy(psi0)

	  for step=1:nsteps

		  dt = dt0
		  accdt = 0.
      tj = 0.

		  while accdt < dt0

			  # propagate one time-step
			  eff_prop = QuFixedStepPropagator( -im*heff, dt)
			  psi1 = propagate(eff_prop, psi)

    		if abs(norm(psi1) - eps) < jtol
    			println("jump at t=$(tj + accdt + dt), norm=$(norm(psi1)), eps=$eps")

    			PnS = 0.
    			for cind = 1:length(c)
    				PnS += real(expectationval(psi1, c[cind]'*c[cind]))
    		  end

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
    			psi = QuStateVec( cop*psi1.elem )
    			normalize!(psi)

    			eps = rand() # draw new random number from [0,1)
    			accdt = accdt + dt
    			dt = dt0 - accdt

    		elseif norm(psi1) < eps
    			dt = dt/2		# jump in the current interval
    		else
    			psi = copy(psi1)  # no jump
    			psi = psi1			  # no jump
    			accdt = accdt + dt
    			dt = dt0 - accdt
    		end

    		println("dt = $dt, norm = $(norm(psi1)), $(norm(psi))")
    	end

      for nop = 1:nops
        exvals[step, nop] = exvals[step, nop] + expectationval(normalize(psi), ops[nop])/ntraj
      end

    	tj = tj + dt0
    end # time-steps

    normalize!(psi)
    rho = rho + QuState(psi)/ntraj

  end  # trajectories

  return rho, exvals
end
