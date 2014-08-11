###############################################################################
# QuDOS : Quantum Dynamics of Open Systems
#
#
###############################################################################
module QuDOS

# exports
export AbstractQuState, QuStateVec, QuState

export QuFixedStepPropagator,
			 QuKrylovPropagator

export propagate

export LindbladQME

export stationary, superop, eff_hamiltonian


# special states
export fockstatevec,
	   	coherentstatevec

# state properties
import Base: norm, getindex, copy, squeeze, *, +, -, /

export purity,
		 ispure,
	   populations,
	   fidelity,
	   ptrace,
	   ptranspose,
	   negativity

# state operations
export normalize, normalize!,
       displace,
	   	expectationval,
	   	applyop,
			 tensor


# operators
export creationop,
	   displacementop,
	   squeezingop,
	   positionop,
	   momentumop

###############################################################################
# Krylov subspace propagator
include("expmv.jl")

###############################################################################
# some special operators
include("operators.jl")

###############################################################################
# state types
abstract AbstractQuState

type QuStateVec <: AbstractQuState
	elem::AbstractMatrix{Complex128}
	nb::Int							 # (total) number of basis functions
	subnb::Vector{Int} 	 # number of basis functions in each
												# subspace

	function QuStateVec( elem::AbstractMatrix{Complex128}, subnb::Vector{Int} )
		nb = size(elem,1) # size of basis
	    if size(elem,2) != 1
	        error("expected column vector, got #dim2=$(size(elem,2))")
	    end
		if prod(subnb) != nb
			error("subspace size mismatch")
		end

		new( elem, nb, subnb )
	end

end

QuStateVec( elem::AbstractMatrix{Complex128} ) = QuStateVec(elem, [size(elem,1)])

type QuState <: AbstractQuState
	elem::AbstractMatrix{Complex128}
	nb::Int				# (total) number of basis functions
	subnb::Vector{Int} 	# number of basis functions in each
						# subspace

	function QuState( elem::AbstractMatrix{Complex128}, subnb::Vector{Int} )
		nb = size(elem,1) # size of basis
	    if nb != size(elem,2)
	        error("expected square matrix")
	    end

		if prod(subnb) != nb
			error("subspace size mismatch")
		end

		new( reshape(elem, nb*nb, 1), nb, subnb )
	end
end

QuState( elem::AbstractMatrix{Complex128} ) = QuState(elem, [size(elem,1)])
QuState( vec::QuStateVec ) = QuState( vec.elem*vec.elem', vec.subnb )

# copy states and state vectors
copy( vec::QuStateVec ) = QuStateVec( copy(vec.elem), copy(vec.subnb) )
copy( rho::QuState ) = QuState( reshape(copy(rho.elem), rho.nb, rho.nb), copy(rho.subnb) )


# access to state matrix elements
getindex( rho::QuStateVec, i::Int ) = rho.elem[ i ]
getindex( rho::QuState, i::Int, j::Int ) = rho.elem[ (j-1)*rho.nb + i, 1]

# apply operator to state vector
applyop( vec::QuStateVec, M::AbstractMatrix) = QuStateVec( M * (vec.elem) )

*( M::AbstractMatrix, vec::QuStateVec) = applyop(vec, M)

# scaling of states and sums
function *(s::Number, qs::AbstractQuState)
	cqs = copy(qs)
	scale!(cqs.elem, s)

	return cqs
end

*(qs::AbstractQuState, s::Number) = *(s,qs)
/(qs::AbstractQuState, s::Number) = *(1/s,qs)

function +(vec1::QuStateVec, vec2::QuStateVec)
	if vec1.subnb != vec2.subnb
		error("Quantum states are not compatible and cannot be added.")
	end

	QuStateVec( (vec1.elem+vec2.elem), copy(vec1.subnb) )
end

function +(rho1::QuState, rho2::QuState)
	if rho1.subnb != rho2.subnb
		error("Quantum states are not compatible and cannot be added.")
	end

	QuState( reshape(rho1.elem+rho2.elem,rho1.nb,rho1.nb),copy(rho1.subnb) )
end

function -(vec1::QuStateVec, vec2::QuStateVec)
	if vec1.subnb != vec2.subnb
		error("Quantum states are not compatible and cannot be subtracted.")
	end

	QuStateVec( (vec1.elem-vec2.elem), copy(vec1.subnb) )
end

function -(rho1::QuState, rho2::QuState)
	if rho1.subnb != rho2.subnb
		error("Quantum states are not compatible and cannot be subtracted.")
	end

	QuState( reshape(rho1.elem-rho2.elem,rho1.nb,rho1.nb), copy(rho1.subnb) )
end


# tensor products of state vectors and states
function tensor( a::QuStateVec, bs::QuStateVec... )

	elem = copy(a.elem)
	subnb = copy(a.subnb)

	for b in bs

		if length(b.subnb) > 1
			error("product with a state vector with more than one subsystem currently not supported")
		end

		elem = kron(elem, b.elem)
		subnb = [subnb, b.nb]
	end

	return QuStateVec( elem, subnb )
end

function tensor( a::QuState, bs::QuState... )

	elem = reshape(a.elem,a.nb,a.nb)
	subnb = copy(a.subnb)

	for b in bs

		if length(b.subnb) > 1
			error("product with a state with more than one subsystem currently not supported")
		end

		elem = kron(elem, reshape(b.elem,b.nb,b.nb))
		subnb = [subnb, b.nb]
	end

	return QuState( elem, subnb )
end

function tensor{T<:AbstractQuState}(a::T, ns::Int)
	states = Array(T, ns)
	fill!(states, a)
	return tensor(states...)
end


###############################################################################
# (sparse) vector representation of a fock state vector
#
function fockstatevec( nb::Int, n::Int )

	return QuStateVec( sparsevec( [n], [Complex(1.)], nb ) )

end

###############################################################################
# (sparse) vector representation of a coherent state vector
#
function coherentstatevec( nb::Int, alpha::Union(Complex128, Float64) )
	# start from ground state
	vec = fockstatevec( nb, 1 )
	# displace to location alpha
	vec = displace( vec, alpha)
	# return normalized vector
	return normalize!( vec )
end

###############################################################################
# get norm of state or state vector
#
function norm( vec::QuStateVec )

	return vecnorm( vec.elem )
end

function norm( rho::QuState )

	return trace( reshape(rho.elem,rho.nb,rho.nb) )
end

# purity of state vectors and states
purity( vec::QuStateVec ) = norm(vec)*norm(vec)

function purity( rho::QuState )

	rhom = reshape(rho.elem,rho.nb,rho.nb)
	return trace( rhom*rhom )
end

ispure(qs::AbstractQuState) = real(purity(qs)) == 1


# extract populations (occupation probabilities)
populations( vec::QuStateVec ) = abs2( vec.elem )
populations( rho::QuState ) = real( diag( reshape(rho.elem,rho.nb,rho.nb) ) )


# fidelity of two pure states is given by the overlap
fidelity( vec1::QuStateVec, vec2::QuStateVec ) = real( abs(vec1.elem'*vec2.elem) )

function fidelity( vec1::QuStateVec, sigma::QuState )
	sigmam = reshape(sigma.elem,sigma.nb,sigma.nb)
	# TODO: use sparse sqrtm
	return real( trace( sqrtm( full(vec1.elem'*sigmam*vec1.elem) ) ) )
end

# fidelity is symmetric
fidelity( sigma::QuState, vec1::QuStateVec ) = fidelity( vec1, sigma)

# fidelity for two states
function fidelity( rho::QuState, sigma::QuState )
	sigmam = full(reshape(sigma.elem,sigma.nb,sigma.nb))
	sqrhom = sqrtm( full(reshape(rho.elem,rho.nb,rho.nb) ) )
	# TODO: use sparse sqrtm
	return real( trace( sqrtm( sqrhom*sigmam*sqrhom ) ) )
end

###############################################################################
# partial trace: trace out subsystems given by osub
#
# The function ptrace is adapted from Toby Cubitt's TrX
# found at http://www.dr-qubit.org/matlab.php which is released under
# GPL license, version 2.
#
function ptrace(rho::QuState, osub::Vector{Int})
	# if there is only one subspace, ptrace is just the norm
	nsub = length(rho.subnb)
	if  nsub == 1
		return norm(rho)
	end
	if length(osub) > nsub || minimum(osub) < 1 || maximum(osub) > nsub
		error("requested subsystems do not match the state")
	end

	rsubnb = reverse(rho.subnb)
	keep   = setdiff([1:nsub],osub) # subs to keep
	nout  = prod(rho.subnb[osub])   # nb of subs to trace out
	nkeep = prod(rho.subnb[keep])   # nb of subs to keep

	# the reshape & permute magic is due to Toby Cubitt
	# see his implementation TrX(...) at http://www.dr-qubit.org/
	#
	# the idea is the following: first the matrix is reshaped
	# such that one gets nsub rows and nsub columns (one pair for each subspace),
	# then all dimensions to be traced out are permuted to the end,
	# the latter are flattened into one dimension, which can be summed over
	perm = nsub+1 .- [keep[end:-1:1]; (keep[end:-1:1].-nsub); osub; (osub.-nsub)]
	x = reshape( permutedims( reshape( full(rho.elem), tuple([rsubnb, rsubnb]...)), perm), (nkeep,nkeep,nout^2))
	return QuState( sparse( squeeze( sum( x[ :,:, [1:nout+1:nout^2]], 3), 3)), rho.subnb[keep] )
end

function ptrace(vec::QuStateVec, osub::Vector{Int})
	# if there is only one subspace, ptrace is just the norm
	if length(vec.subnb) == 1
		return norm(vec)
	end

	# ptrace of a pure state is typically a mixed state
	return ptrace( QuState(vec), osub)
end

# partial transpose: transpose only for given subsystem sub
# note: the resulting matrix is not necessarily a proper quantum state
#
# The functions ptranspose is adapted from Toby Cubitt's Tx
# found at http://www.dr-qubit.org/matlab.php which is released under
# GPL license, version 2.
#
function ptranspose(rho::QuState, sub::Int)
	# if there is only one subspace, ptrace is just the transpose of rho
	nsub = length(rho.subnb)
	if  nsub == 1
		return QuState( transpose(reshape(rho.elem,rho.nb,rho.nb)), rho.subnb)
	end
	if sub < 1 || sub > nsub
		error("requested subsystem does not match the state")
	end

	# the idea is again due to to Toby Cubitt
	# see his implementation Tx(...) at http://www.dr-qubit.org/
	rsubnb = reverse(rho.subnb)
	perm = [1:2*nsub]
	perm[(nsub+1-sub)], perm[(2*nsub+1-sub)] = perm[(2*nsub+1-sub)], perm[(nsub+1-sub)]
	x = reshape( permutedims( reshape( full(rho.elem), tuple([rsubnb rsubnb]...) ), perm), rho.nb, rho.nb)
	return QuState( sparse(x), rho.subnb )
end

# negativity gives a measure of entanglement
# (see Vidal & Werner, PRA65, 032314 (2002))
#
function negativity(rho::QuState, sub::Int)
	ptrho = ptranspose(rho, sub)
	# TODO: use a proper method to diagonalize for
	#       sparse matrices (in v0.2,0.3 there is an error when calling eigvals for sparse)
	mu = eigvals( full(reshape(ptrho.elem, ptrho.nb, ptrho.nb)) )
	# negativity is the sum of the negative eigenvalues
	return abs( sum( mu[ find(x->x<0, real(mu)) ] ) )
end

###############################################################################
#
function expectationval( vec::QuStateVec, o::AbstractMatrix )

	return (vec.elem'*o*vec.elem)[1]
end

function expectationval( rho::QuState, o::AbstractMatrix )

	return trace( o*reshape(rho.elem,rho.nb,rho.nb))
end


###############################################################################
# normalize states and state vectors
function normalize( vec::QuStateVec )
	normalize!(copy(vec))
end

function normalize( rho::QuState )
	normalize!(copy(rho))
end

function normalize!( vec::QuStateVec )
	scale!( vec.elem, 1/sqrt( (vec.elem'*vec.elem)[1] ) )
	return vec
end

function normalize!( rho::QuState )
	rhom =  reshape(rho.elem,rho.nb,rho.nb)
	scale!( rho.elem, 1/trace(rhom) )
	return rho
end


###############################################################################
# displace state vector in phase space
function displace( psi::QuStateVec, alpha::Union(Complex128, Float64) )
	# displace by alpha in phase space
	# TODO: use sparse matrix exponential if required
	ad = alpha*creationop( size(psi.elem,1) )

	QuStateVec( convert(typeof(psi.elem), expm( full(ad - ad') )*psi.elem) )
end

###############################################################################
# squeeze state vector in phase space
function squeeze( psi::QuStateVec, z::Union(Complex128, Float64) )
	ad2 = 0.5*z*creationop( size(psi,1) )*creationop( size(psi.elem,1) )
	# TODO: use sparse matrix exponential if required
	QuStateVec( convert(typeof(psi.elem), expm( full(ad2' - ad2) )*psi.elem) )
end

###############################################################################
# propagators
include("propagate.jl")

# QME types and functionality
include("qme.jl")

# MCWF method (aka quantum jump)
include("mcwf.jl")

end # module
