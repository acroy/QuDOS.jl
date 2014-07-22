## QuDOS
## interface for quantum master equations

# 
include("nulls.jl")

# abstract super-type for all QMEs
abstract AbstractQuMasterEquation


# QMEs of Lindblad type defined by
#    L rho = -im*[h, rho] - 1/2 sum^Nl_l ( c^dag_l c_l rho + rho c^dag_l c_l )
#                         +     sum^Nl_l ( c_l rho c^dag_l )
#
abstract AbstractLindbladQME <: AbstractQuMasterEquation


type LindbladQME <: AbstractLindbladQME
    hamop::AbstractMatrix{Complex128}			# Hamilton operator
    lbops::Vector{AbstractMatrix{Complex128}}		# Lindblad operators

#    lb::AbstractMatrix{Complex128}			# cached Lindblad super-operator ?

end

# "add" two LindbladQMEs by adding the respective Hamiltonia and joining the Lindblad operators
+(qme1::LindbladQME, qme2::LindbladQME) = LindbladQME( qme1.hamop + qme2.hamop, vcat(qme1.lbops, qme2.lbops))

# return super-operator matrix
function superop(qme::LindbladQME)

     lb = nothing
     if nprocs() > 1
         lb = plindbladsuperop( qme.hamop, qme.lbops)
     else
         lb = lindbladsuperop( qme.hamop, qme.lbops)
     end

    return lb
end

# return effective (non-hermitian) Hamiltonian
function eff_hamiltonian(qme::AbstractLindbladQME)
    nlop = length(qme.lbops)
    heff = qme.hamop
    for l=1:nlop
       heff = heff - im*0.5*qme.lbops[l]'*qme.lbops[l]
    end

    return heff
end

# find stationary state, starting from rho
# TODO: for some Lindblad ops this doesn't converge
#       and it seems not to be reliable
function stationary(qme::AbstractLindbladQME, rho::QuState; precond=true, tol=1e-5)

    lb = superop(qme)

    presyl=nothing
    if precond
        syl = sylvsuperop(qme.hamop, qme.lbops)
        presyl = lufact(syl)
    end

    v = full(rho.elem)[:]
    v, avnorm, iter = nulls!(v, lb, presyl)

    return normalize!( QuState(reshape(v,rho.nb,rho.nb), rho.subnb))
end

# constructors for propagator types
QuFixedStepPropagator(qme::AbstractLindbladQME, dt::Real) = QuFixedStepPropagator(superop(qme), dt::Real)
QuKrylovPropagator(qme::AbstractLindbladQME) = QuKrylovPropagator(superop(qme))
