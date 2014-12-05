# TODO: if possible hook up with  IterativeSolvers.jl

###############################################################################
# calculate matrix exponential acting on some vector, w = exp(t*A)*v,
# using the Krylov subspace approximation
#
# see R.B. Sidje, ACM Trans. Math. Softw., 24(1):130-156, 1998
# and http://www.maths.uq.edu.au/expokit
#
# The original EXPOKIT comes with the following copyright notice:
#
# Permission to use, copy, modify, and distribute EXPOKIT and its
# supporting documentation for non-commercial purposes, is hereby
# granted without fee, provided that this permission message and
# copyright notice appear in all copies. Approval must be sought for
# commercial purposes as testimony of its usage in applications.
#
# Neither the Institution (University of Queensland) nor the Author
# make any representations about the suitability of this software for
# any purpose.  This software is provided ``as is'' without express or
# implied warranty.
#
# The work resulting from EXPOKIT has been published in ACM-Transactions
# on Mathematical Software, 24(1):130-156, 1998.
#
expmv{T}( vec::Vector{T}, t::Number, amat::AbstractMatrix, tol::Real=1e-7, m::Int=min(30,size(amat,1))) = expmv!(copy(vec), t, amat, tol, m)

function expmv!{T}( vec::Vector{T}, t::Number, amat::AbstractMatrix, tol::Real=1e-7, m::Int=min(30,size(amat,1)))

	if size(vec,1) != size(amat,2)
		error("dimension mismatch")
	end

	# safety factors
	gamma = 0.9
	delta = 1.2

	btol = 1e-7 	# tolerance for "happy-breakdown"
	maxiter = 10	# max number of time-step refinements

	anorm = norm(amat, Inf)
	rndoff= anorm*eps()

	# estimate first time-step and round to two significant digits
	beta = norm(vec)
	r = 1/m
	fact = (((m+1)/exp(1.))^(m+1))*sqrt(2.*pi*(m+1))
	tau = (1./anorm)*((fact*tol)/(4.*beta*anorm))^r
	tau = signif(tau, 2)

	vm = similar(vec, size(vec,1), m+1)
	hm = zeros(T,m+2,m+2)

	tf = abs(t)
	tsgn = sign(t)
	tk = zero(tf)

	v = vec
	mx = m
	while tk < tf
		tau = min(tf-tk, tau)

		# Arnoldi procedure
		vm[:,1] = v/beta
		mx = m
		for j=1:m
			p = amat*vm[:,j]
			for i=1:j
				hm[i,j] = dot(vm[:,i], p[:,1])
				p = p - hm[i,j]*vm[:,i]
			end
			hm[j+1,j] = norm(p)

			if real(hm[j+1,j]) < btol	# happy-breakdown
				tau = tf - tk
				err_loc = btol

				F = expm(tsgn*tau*hm[1:j,1:j])
				v[:] = beta*vm[:,1:j]*F[1:j,1]

				mx = j
				break
			end

			vm[:,j+1] = p/hm[j+1,j]
		end
		hm[m+2,m+1] = one(T)
		avnorm = norm(amat*vm[:,m+1])

		# propagate using adaptive step size
		iter = 1
		while (iter < maxiter) && (mx == m)

			F = expm(tsgn*tau*hm[1:m+2,1:m+2])

			# local error estimation
			err1 = abs( beta*F[m+1,1] )
			err2 = abs( beta*F[m+2,1] * avnorm )

			if err1 > 10*err2	# err1 >> err2
				err_loc = err2
				r = 1/m
			elseif err1 > err2
	 	 	err_loc = (err1*err2)/(err1-err2)
	 	 	r = 1/m
    	else
				err_loc = err1
				r = 1/(m-1)
			end

			# time-step sufficient?
			if err_loc <= delta * tau * (tau*tol/err_loc)^r
				v[:] = beta*vm[:,1:m+1]*F[1:m+1,1]
				break
			end
			tau = gamma * tau * (tau*tol/err_loc)^r		# estimate new time-step
			tau = signif(tau, 2)				# round to 2 signiﬁcant digits
																	# to prevent numerical noise
			iter = iter + 1
		end
		if iter == maxiter
			# TODO, here an exception should be thrown, but which?
			error("Number of iterations exceeded $(maxiter). Requested tolerance might be to high.")
		end

		beta = norm(v)
		tk = tk + tau

		tau = gamma * tau * (tau*tol/err_loc)^r		# estimate new time-step
		tau = signif(tau, 2)			# round to 2 signiﬁcant digits
															# to prevent numerical noise
		err_loc = max(err_loc,rndoff)
	end

	return v
end


function test_expmv(n::Int)

	A = sprand(n,n,0.4)
	v = eye(n,1)[:]

	tic()
	w1 = expmv(v, 1.0, A)
	t1 = toc()

	tic()
	w2 = expm(full(A))*v
	t2 = toc()

	return norm(w1-w2)/norm(w2), t1, t2
end
