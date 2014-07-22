# propagators for QMEs

abstract AbstractQuPropagator


# fixed step propagator, which calculates the
# matrix exponential expm(A*dt) for a
# given matrix A and time-step dt
# (potentially very slow!)
type QuFixedStepPropagator <: AbstractQuPropagator
	U::AbstractMatrix{Complex128}
	dt::Union(Real,Complex)

	function QuFixedStepPropagator( A::AbstractMatrix, dt::Union(Real,Complex))
		if size(A,1) != size(A,2)
	  	error("expected square matrix")
		end

		new( expm( full(A)*dt ), dt )
	end
end

# propagate a state or state vector using a fixed-step propagator
function propagate( prop::QuFixedStepPropagator, vec::QuStateVec; tspan=[0., prop.dt] )
	v = copy(vec.elem)

	nsteps = convert(Int, div(tspan[end],prop.dt))
	for step=1:nsteps
		v = prop.U*v
	end

	QuStateVec( convert(typeof(vec.elem), v), vec.subnb )
end

function propagate( prop::QuFixedStepPropagator, rho::QuState; tspan=[0., prop.dt] )
	v = copy(rho.elem)

	nsteps = convert(Int, div(tspan[end],prop.dt))
	for step=1:nsteps
		v = prop.U*v
	end

	QuState( reshape(convert(typeof(rho.elem), v), rho.nb, rho.nb), rho.subnb )
end

###############################################################################
# propagator using expmv (Krylov subspace propagation)

type QuKrylovPropagator <: AbstractQuPropagator
	A::AbstractMatrix{Complex128}

	function QuKrylovPropagator( A::AbstractMatrix)
	    if size(A,1) != size(A,2)
	        error("expected square matrix")
	    end

		new( A )
	end
end

function propagate( prop::QuKrylovPropagator, rho::QuState; tspan=[0., 1.] )
	if length(tspan) < 2
		error("expected vector of evaluation times")
	end

	v = full(copy(rho.elem))[:]

	for ts=2:length(tspan)

		v = expmv!(v, tspan[ts]-tspan[ts-1], prop.A)
	end

	QuState( reshape(convert(typeof(rho.elem), reshape(v,length(v),1)), rho.nb, rho.nb), rho.subnb )
end


###############################################################################
