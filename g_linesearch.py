# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:42:50 2023

@author: Sani Salisu
"""


# BetaTypes of the conjugate gradient method in pymanopt was changed.
#BetaTypes = Enum("BetaTypes", "DaiYuan PolakRibiere Hybrid1 Hybrid2".split())
import numpy as np
class BackTrackingLineSearcher:
    """Back-tracking line-search algorithm."""

    def __init__(
        self,
        contraction_factor=0.5,
        optimism=2,
        sufficient_decrease=1e-4,
        max_iterations=25,
        initial_step_size=1,
    ):
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.sufficient_decrease = sufficient_decrease
        self.max_iterations = max_iterations
        self.initial_step_size = initial_step_size

        self._oldf0 = None

    def search(self, objective, manifold, x, d, f0, df0, gradient):
        """Function to perform backtracking line search.

        Args:
            objective: Objective function to optimize.
            manifold: The manifold to optimize over.
            x: Starting point on the manifold.
            d: Tangent vector at ``x``, i.e., a descent direction.
            df0: Directional derivative at ``x`` along ``d``.

        Returns:
            A tuple ``(step_size, newx)`` where ``step_size`` is the norm of
            the vector retracted to reach the suggested iterate ``newx`` from
            ``x``.
        """
        # Compute the norm of the search direction
        norm_d = manifold.norm(x, d)

        if self._oldf0 is not None:
            # Pick initial step size based on where we were last time.
            alpha = 2 * (f0 - self._oldf0) / df0
            # Look a little further
            alpha *= self.optimism
        else:
            alpha = self.initial_step_size / norm_d
        alpha = float(alpha)

        # Make the chosen step and compute the cost there.
        newx = manifold.retraction(x, alpha * d)
        newf = objective(newx)
        step_count = 1

        # Backtrack while the Armijo criterion is not satisfied
        while (
            newf > f0 + self.sufficient_decrease * alpha * df0
            and step_count <= self.max_iterations
        ):

            # Reduce the step size
            alpha = self.contraction_factor * alpha

            # and look closer down the line
            newx = manifold.retraction(x, alpha * d)
            newf = objective(newx)

            step_count = step_count + 1

        # If we got here without obtaining a decrease, we reject the step.
        if newf > f0:
            alpha = 0
            newx = x

        step_size = alpha * norm_d

        self._oldf0 = f0

        return step_size, newx


class AdaptiveLineSearcher:
    """Adaptive line-search algorithm."""

    def __init__(
        self,
        contraction_factor=0.5,
        sufficient_decrease=0.5,
        max_iterations=10,
        initial_step_size=1,
    ):
        self._contraction_factor = contraction_factor
        self._sufficient_decrease = sufficient_decrease
        self._max_iterations = max_iterations
        self._initial_step_size = initial_step_size
        self._oldalpha = None

    def search(self, objective, manifold, x, d, f0, df0, gradient):
        norm_d = manifold.norm(x, d)

        if self._oldalpha is not None:
            alpha = self._oldalpha
        else:
            alpha = self._initial_step_size / norm_d
        alpha = float(alpha)

        newx = manifold.retraction(x, alpha * d)
        newf = objective(newx)
        cost_evaluations = 1

        while (
            newf > f0 + self._sufficient_decrease * alpha * df0
            and cost_evaluations <= self._max_iterations
        ):
            # Reduce the step size.
            alpha *= self._contraction_factor

            # Look closer down the line.
            newx = manifold.retraction(x, alpha * d)
            newf = objective(newx)

            cost_evaluations += 1

        if newf > f0:
            alpha = 0
            newx = x

        step_size = alpha * norm_d

        # Store a suggestion for what the next initial step size trial should
        # be. On average we intend to do only one extra cost evaluation. Notice
        # how the suggestion is not about step_size but about alpha. This is
        # the reason why this line search is not invariant under rescaling of
        # the search direction d.

        # If things go reasonably well, try to keep pace.
        if cost_evaluations == 2:
            self._oldalpha = alpha
        # If things went very well or we backtracked a lot (meaning the step
        # size is probably quite small), speed up.
        else:
            self._oldalpha = 2 * alpha

        return step_size, newx


class LineSearchWolfe:
    def __init__(self, c1: float=1e-4, c2: float=0.9):
        self.c1 = c1
        self.c2 = c2

    def __str__(self):
        return 'Wolfe'

    def search(self, objective, man, x, d, f0, df0, gradient):
        '''
        Returns the step size that satisfies the strong Wolfe condition.
        Scipy.optimize.line_search in SciPy v1.4.1 modified to Riemannian manifold.

        ----------
        References
        ----------
        [1] SciPy v1.4.1 Reference Guide, https://docs.scipy.org/
        '''
        fc = [0]
        gc = [0]
        gval = [None]
        gval_alpha = [None]

        def phi(alpha):
            fc[0] += 1
            return objective(man.retraction(x, alpha * d))

        def derphi(alpha):
            newx = man.retraction(x, alpha * d)
            newd = man.transport(x, newx, d)
            gc[0] += 1
            gval[0] = gradient(newx)  # store for later use
            gval_alpha[0] = alpha
            return man.inner_product(newx, gval[0], newd)

        gfk = gradient(x)
        derphi0 = man.inner_product(x, gfk, d)
        stepsize = _scalar_search_wolfe(phi, derphi, self.c1, self.c2, maxiter=100)
        
        
        if stepsize is None:
            stepsize = 1e-6
        
        newx = man.retraction(x, stepsize * d)
        
        return stepsize, newx


def _scalar_search_wolfe(phi, derphi, c1=1e-4, c2=0.9, maxiter=100):
    phi0 = phi(0.)
    derphi0 = derphi(0.)
    alpha0 = 0
    alpha1 = 1.0
    phi_a1 = phi(alpha1)
    phi_a0 = phi0
    derphi_a0 = derphi0
    for i in range(maxiter):
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2)
            break

        derphi_a1 = derphi(alpha1)
        #if (abs(derphi_a1) <= c2 * abs(derphi0)):
        if (derphi_a1) >= c2 *(derphi0):
          # if (abs(derphi_a1) <= c2 * abs(derphi0)):
          # the commented line is SWC
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1
    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        print('The line search algorithm did not converge')
    
    return alpha_star


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2):
    """
    Part of the optimization algorithm in `_scalar_search_wolfe`.
    """
    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi
        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha
        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= c2 * abs(derphi0):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin