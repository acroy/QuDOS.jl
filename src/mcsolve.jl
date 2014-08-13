# mcsolve method for Monte Carlo Wavefunctions. It has a similar constructure as the QuTIP mcsolve function.

function mcsolve(H::Union(SparseMatrixCSC{Complex{Float64},Int64},AbstractMatrix{Complex128}),
                 psi0::Array{Complex128,1},
                 tlist::Array{Float64,1}, ops::Vector{AbstractMatrix{Complex128}}, measms::Array{AbstractMatrix{Complex128},1},
                 ntraj::Union(Int64,Array{Int64,1})=500 )
  # General mcsolve method with normal data type inputs.
  #
  # inputs: H::AbstractMatrix{Complex128}. - The Hamiltonian for the system.
  #         psi0::Vector{Complex128}.      - The initial state of the system.
  #         tlist::Vector.                 - Time points as a vector.
  #         ops::Vector{AbstractMatrix{Complex128}}. - The Lindblad operators or jump operators as a vector with matrix elements.
  #         measms::Vector{AbstractMatrix{Complex128}}. - The measurement operators.
  #         ntraj::Vector{Int}=500.        - Number of trajectories for averaging. It can be a vector with increasing integers.
  #
  # outputs:
  #          exvals=Array(Float64, lengthtraj, nmeasm, nsteps)      - Expectation values of the measurement operators.
  #                  It is given in a matrix form, where lengthtraj is the number of the recorded trajectories, nmeasm is the
  #                  measurement operators, and nsteps is the number of the tlist time points.
  #

  # BEtter to uniformality define new types: 1. QuDOSResult{K<:Float64,OP<:Complex128,N<:Int64}-->StateVec,Ops,Expect
  #                                          2. QOperators

  #Constants.
  hbar=1 # Should be defined as the real value, right?

  nsteps=length(tlist)  # Number of time steps.
  nops = length(ops)    # Number of jump operators.
  nd = size(ops[1],1)   # By default, all operators should be represented as square matrices.
  nmeasm=length(measms) # Number of measurement operators.
  lengthtraj=length(ntraj)   # Number of averaging points for given trajectory numbers.
  exvals = zeros(Float64, lengthtraj, nmeasm,nsteps)   # Expection values at each time step for the given measurement operators for given number of trajectories.
  exvals_temp = zeros(Float64, nmeasm,nsteps)   # Expection values at each time step for the given measurement operators in accumulated trajectories.

  # Get information of Hamiltonians and jump operators.
  # c = Vector{AbstractMatrix{Complex128}} # Initialize the jump operators.
  # c = ops #linbladops(qme)   # Jump operators.
  heff = H   # Initialize the effective Hamiltonians.
  for i=1:nops
    heff = heff - 0.5im*hbar*ops[i]'*ops[i]  #eff_hamiltonian(qme)
  end

  # rho = QuState( zeros(Complex128, psi0.nb, psi0.nb), psi0.subnb )

  # Loop over all trajectories.
  maxntraj=ntraj[end]
  trajind=1   # Counting readout index in the ntraj vector.
  for traj=1:maxntraj # Loop over all trajectories and including all given stopping points.
	  println("Starting trajectory $traj/$maxntraj:")
    # The initial state and measurement records.
	  eps = rand() # Draw a random number from [0,1) to represent the probability that a quantum jump occurs.
    # psi = Vector{AbstractVector{Complex128}}(nsteps) # Initialize the state for each trajectory.
	  psi  = copy(psi0)   # Initialize the quantum state vector after each evolution step.
    psi1  = copy(psi0)  # Initialize the quantum state vector before evolution.
    psi_norm = 1 # Initialize the norm of the state vector.
    t0=tlist[1]   # Assme tlist has a length larger than 1.
    for nm=1:nmeasm
      exvals_temp[nm,1] += real(psi'*measms[nm]*psi)[1]
    end
    # All the rest time steps.
	  for step=2:nsteps # Step over all time steps.
		  dt = tlist[step]-t0
		  if psi_norm > eps # No jump.
			  # Propagate one time-step until the jumping condition satisfies.
			  #eff_prop = QuFixedStepPropagator( -1im*heff/hbar, dt)
			  psi1 = expmv(psi,dt,-1im*heff/hbar) # propagate(eff_prop, psi)
    		println("t = $(tlist[step]), norm = $(norm(psi1)), evolved from $(norm(psi)).")
        psi=copy(psi1)   # State for the next step after the evolution time dt.
      else  # Otherwise, jump happens.
    			println("Jump at t=$(tlist[step]), norm=$(norm(psi1)), eps=$eps .")
          # Calculate the probability distribution for each jump operator.
    			Pi = zeros(Float64,nops)   # Initialize the probability of each jump.
          PnS = 0.   # Initialize the unnormalized accumulated probability of jumps.
          # dp = 0.   # Initialize the accumulated probability of jumps.
    			for cind = 1:nops
            Pi[cind] = norm(ops[cind]*psi1).^2 # real(psi1'*ops[cind]'*ops[cind]*psi1)
            PnS += Pi[cind]   # Total probability of jumps without normalization.
    		  end
          # Judge which jump.
          Pn = 0.  # Initialize the normalized accumulated probability of jumps.
    			cop = eye(size(heff,1))   # Initialize the jump operator.
    			for cind = 1:nops
    				Pn += Pi[cind]/PnS
    				println("Pn($cind)=$Pn")
    				if Pn >= eps
    					println("Using jump operator $cind")
    					cop=ops[cind]
    					break
    				end
    			end
    			# New state after the jump of cop.
    			psi1 = cop*psi #QuStateVec( cop*psi1.elem )
          # psi = QuStateVec(psi1)
    			psi = psi1./norm(psi1) # normalize!(psi1)
          # Generate a new random number after the jump.
    			eps = rand() # draw new random number from [0,1)
    		end
      psi_norm=norm(psi)   # Norm after this step of evolution.
      t0=tlist[step] # Record time to get dt for the next iteration.
      # Expectation values of measurement operators.
      for nm = 1:nmeasm
        exvals_temp[nm,step] += real(psi'*measms[nm]*psi)[1]
      end

    end # Time-steps

    # Averaging expectation values for requested quantum trajectory numbers.
    if traj==ntraj[trajind]
       exvals[trajind,:,:] = exvals_temp[:,:]./traj
       trajind += 1
    end
  end  # trajectories

  return exvals
end

######################################################################################
# Other input type patterns.
###################################################################################3##
function mcsolve(qme::LindbladQME, psi0::AbstractQuState,
                 tlist::Array{Float64,1}, measms::Array{AbstractArray{Complex128,2},1},
                 ntraj::Array{Int,1}=[500] )
  # mcsolve method with AbstractLindbladQME and AbstractQuState types of inputs.

  H = qme.hamop
  psi_init = psi0.elem
  ops = QuDOS.linbladops(qme)
  exvals = mcsolve(H, psi0, tlist, ops, measms, ntraj)
  return exvals
end

function mcsolve(qme::LindbladQME, psi0::AbstractQuState,
                 tlist::Array{Float64,1}, measms::Array{AbstractArray{Complex128,2},1},
                 ntraj::Int64=500 )
  # mcsolve method with AbstractLindbladQME and AbstractQuState types of inputs.

  H = qme.hamop
  psi_init = vec(full(psi0.elem))
  ops = QuDOS.linbladops(qme)
  exvals = mcsolve(H, psi_init, tlist, ops, measms, ntraj)
  return exvals
end

function mcsolve(qme::LindbladQME, psi0::AbstractQuState,
                 tlist::Array{Float64,1}, measms::Array{AbstractArray{Complex128,2},1},
                 ntraj::Array{Int64,1}=[500] )
  # mcsolve method with AbstractLindbladQME and AbstractQuState types of inputs.

  H = qme.hamop
  psi_init = vec(full(psi0.elem))
  ops = QuDOS.linbladops(qme)
  exvals = mcsolve(H, psi_init, tlist, ops, measms, ntraj)
  return exvals
end

function mcsolve(H::SparseMatrixCSC{Complex{Float64},Int64}, psi0::QuStateVec,
                 tlist::Array{Float64,1}, ops::Array{AbstractArray{Complex{Float64},2},1},
                 measms::Array{AbstractArray{Complex128,2},1},
                 ntraj::Int64=500 )
  # mcsolve method with AbstractQuState type of inputs.

  psi_init = vec(full(psi0.elem))
  exvals = mcsolve(H, psi_init, tlist, ops, measms, ntraj)
  return exvals
end
