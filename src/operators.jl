## QuDOS
## Operators and Super-Operators

###############################################################################
# (sparse) matrix representation of a bosonic creation operator ad,
# which satisfies the commutation relation
#
#     ad'*ad - ad*ad' = identity
#
function creationop( nb::Int )

	sparse( [2:nb], [1:nb-1], sqrt(linspace( 1, nb-1, nb-1)), nb, nb )

end

###############################################################################
# (sparse) matrix representation of a bosonic displacement operator
#
function displacementop( nb::Int, alpha::Union(Complex128, Float64) )
	ad = alpha*creationop( nb )
	# displacement to location alpha
	# TODO: use proper sparse matrix exponential
	sparse( expm( dense(ad - ad') ) )
end

###############################################################################
# (sparse) matrix representation of a bosonic squeezing operator
#
function squeezingop( nb::Int, z::Union(Complex128, Float64) )
	ad2 = 0.5*z*creationop( nb )*creationop( nb )
	# TODO: use proper sparse matrix exponential
	sparse( expm( dense(ad2' - ad2) ) )
end

###############################################################################
# (sparse) matrix representation of a bosonic position operator
#
function positionop( nb::Int )
	ad = creationop( nb )
	(ad + ad')/sqrt(2.)
end

###############################################################################
# (sparse) matrix representation of a bosonic momentum operator
#
function momentumop( nb::Int )
	ad = creationop( nb )
	im*(ad - ad')/sqrt(2.)
end


###############################################################################
# (sparse) matrix representation of a Lindblad superoperator L, which
# is given in terms of Hamiltonian h and Lindblad operators c_l
#
#    L rho = -im*[h, rho] - 1/2 sum^Nl_l ( c^dag_l c_l rho + rho c^dag_l c_l )
#                         +     sum^Nl_l ( c_l rho c^dag_l )
#
function lindbladsuperop( h::AbstractMatrix, c )
	# number of basis functions
	nb = size(h, 1)

	# construct effective Hamiltonian from
	# one or more Lindblad operators
	nlop = length(c)
	heff = zeros(eltype(h), size(h))
	for l=1:nlop
		heff = heff + 0.5*c[l]'*c[l]
	end

	# construct Lindblad superoperator
	SI = Array(Int,0)
	SJ = Array(Int,0)
	Lvals = Array(Complex128,0)

	for m=1:nb
		for n=1:nb
			for i=1:nb
				for j=1:nb

					sm = (m-1)*nb + n
					sj = (i-1)*nb + j

					lv = zero(Complex128)
					for l=1:nlop
						lv = lv + c[l][m,i]*conj(c[l][n,j])
					end

					if j==n
						lv = lv - im*h[m,i] - heff[m,i]
					end
					if i==m
						lv = lv + im*h[j,n] - heff[j,n]
					end

					if real(lv)!=0 || imag(lv)!=0
						push!(SI, sm)
						push!(SJ, sj)
						push!(Lvals, lv)
					end

				end
			end
		end
	end

	return sparse(SI, SJ, Lvals, nb*nb, nb*nb)
end

function plindbladsuperop( h::AbstractMatrix, c )
	# number of basis functions
	nb = size(h, 1)

	# construct effective Hamiltonian from
	# one or more Lindblad operators
	nlop = length(c)
	heff = zeros(eltype(h), size(h))
	for l=1:nlop
		heff = heff + 0.5*c[l]'*c[l]
	end

	# construct Lindblad superoperator
  # using several processes, each process returns a part of the Linblad matrix
	splb = @parallel (+) for m=1:nb

	    SI = Array(Int,0)
	    SJ = Array(Int,0)
	    Lvals = Array(Complex128,0)

            for n=1:nb
			for i=1:nb
				for j=1:nb

					lv = zero(Complex128)
					for l=1:nlop
						lv = lv + c[l][m,i]*conj(c[l][n,j])
					end

					if j==n
						lv = lv - im*h[m,i] - heff[m,i]
					end
					if i==m
						lv = lv + im*h[j,n] - heff[j,n]
					end

					if real(lv)!=0 || imag(lv)!=0
					    sm = (m-1)*nb + n
					    sj = (i-1)*nb + j

					    push!(SI, sm)
					    push!(SJ, sj)
					    push!(Lvals, lv)
					end

				end
			end
		end

  	sparse(SI, SJ, Lvals, nb*nb, nb*nb)
	end

  return splb
end

###############################################################################
# (sparse) matrix representation of a superoperator S, which
# is given in terms of Hamiltonian h and Lindblad operators c_l
# (non-hermitian Hamiltonian)
#
#    S rho = -im*[h, rho] - 1/2 sum^Nl_l ( c^dag_l c_l rho + rho c^dag_l c_l )
#
function sylvsuperop( h::AbstractMatrix, c )
  # number of basis functions
  nb = size(h, 1)

  # construct effective Hamiltonian from
  # one or more Lindblad operators
  nlop = length(c)
  heff = zeros(eltype(h), size(h))
  for l=1:nlop
    heff = heff + 0.5*c[l]'*c[l]
  end

  # construct Lindblad superoperator
  SI = Array(Int,0)
  SJ = Array(Int,0)
  Lvals = Array(Complex128,0)

  for m=1:nb
    for n=1:nb
      for i=1:nb

        j=n
        lv = -im*h[m,i] - heff[m,i]

        if real(lv)!=0 || imag(lv)!=0
          sm = (m-1)*nb + n
          sj = (i-1)*nb + j

          push!(SI, sm)
          push!(SJ, sj)
          push!(Lvals, lv)
        end
      end

      for j=1:nb
        i=m
        lv = im*h[j,n] - heff[j,n]

        if real(lv)!=0 || imag(lv)!=0
          sm = (m-1)*nb + n
          sj = (i-1)*nb + j

          push!(SI, sm)
          push!(SJ, sj)
          push!(Lvals, lv)
        end


      end
    end
  end

  return sparse(SI, SJ, Lvals, nb*nb, nb*nb)
end
