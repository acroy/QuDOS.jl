# use Arnoldi iteration + SVD to find the eigenvector of a matrix
# which has approximately a zero eigenvalue
#
# arguments: vec  - initial vector for iterations
#            amat - matrix for which the eigenvector is requested
#            pre  - preconditioner
#
# keywords:  tol  - tolerance for convergence, ie, norm(amat*vec) < tol
#            m    - size of Krylov subspace
#            norm - user defined norm
#            maxiter - max number of time-step refinements
#
# returns:   vec, norm(amat*vec), number of iterations
function nulls!( vec, amat, pre=nothing; tol=1e-5,
                             m=min(30,size(amat,1)),
                             norm=Base.norm,
                             maxiter=10000)

  if size(vec,1) != size(amat,2)
    error("dimension mismatch")
  end

  if pre==nothing
    preconditioner(x) = x
  else
    preconditioner(x) = pre\x
  end

  btol = 1e-7 	# tolerance for "happy-breakdown"
  # maxiter -> max number of time-step refinements

  # vm = similar(vec, size(vec,1), m+1)
  vm = Array(typeof(vec), m+1)
  hm = zeros(eltype(vec),m+1,m+1)

  beta = norm(vec)
  v = vec

  avnorm = norm(amat*vec)
  # println("-- norm = $(avnorm)")

  iter = 1
  while true

    # Arnoldi procedure
    vm[1] = v/beta
    mx = m
    for j=1:m
      p = amat*vm[j]
      p = preconditioner(p)
      for i=1:j
        hm[i,j] = dot(vm[i], p[:])
        p = p - hm[i,j]*vm[i]
      end
      hm[j+1,j] = norm(p)

      if real(hm[j+1,j]) < btol	# happy-breakdown
        mx = j
        break
      end

      vm[j+1] = p/real(hm[j+1,j])
    end
    # hm[m+2,m+1] = one(T)

    U,S,V = svd(hm[1:mx,1:mx])

    v = vm[1]*V[1,end]
    for i=2:mx
      v = v + vm[i]*V[i,end]
    end

    avnorm = norm(amat*v)
    # println("-- norm = $(avnorm), svd val = $(S[end]), mx = $(mx)")

    if avnorm < tol
      break
    end
    iter = iter + 1
    if iter == maxiter
      # TODO, here an exception should be thrown, but which?
      error("Number of iterations exceeded $(maxiter). Requested tolerance might be to high.")
    end

    beta = norm(v)
  end

  return v, avnorm, iter
end
