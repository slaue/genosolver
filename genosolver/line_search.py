'''
//    GENO is a solver for non-linear optimization problems.
//    It can solve constrained and unconstrained problems.
//
//    Copyright (C) 2018-2019 Soeren Laue
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Affero General Public License as published
//    by the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Affero General Public License for more details.
//
//    You should have received a copy of the GNU Affero General Public License
//    along with this program. If not, see <http://www.gnu.org/licenses/>.
//
//    Contact the developer:
//
//    E-mail: soeren.laue@uni-jena.de
//    Web:    http://www.geno-project.org
'''
import numpy

def dcstep(stx: float, 
           fx: float, 
           dx: float, 
           sty: float, 
           fy: float, 
           dy: float, 
           stp: float, 
           fp: float,
           dp: float,
           brackt: bool, 
           stpmin: float, 
           stpmax: float, 
           verbose: int,
           np=numpy):

  sgnd = dp * np.sign(dx)
  if (fp > fx):
    theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
    s = max(abs(theta), abs(dx), abs(dp))
    ths = theta / s
    gamma = s * np.sqrt(ths*ths - (dx / s) * (dp / s))
    if (stp < stx): gamma = -gamma
    p = (gamma - dx) + theta
    q = ((gamma - dx) + gamma) + dp
    r = p / q
    stpc = stx + r*(stp - stx)
    stpq = stx + ((dx/((fx - fp)/(stp - stx) + dx))/2.0) * (stp - stx);
    if (abs(stpc-stx) < abs(stpq-stx)):
      stpf = stpc
    else:
      stpf = stpc + (stpq - stpc)/2.0
    brackt = True
  elif (sgnd < 0.0):
    theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
    s = max(abs(theta), abs(dx), abs(dp))
    ths = theta/s
    gamma = s*np.sqrt(ths*ths - (dx/s)*(dp/s))
    if (stp > stx): gamma = -gamma
    p = (gamma - dp) + theta
    q = ((gamma - dp) + gamma) + dx
    r = p/q
    stpc = stp + r*(stx - stp)
    stpq = stp + (dp/(dp - dx))*(stx - stp)
    if (abs(stpc-stp) > abs(stpq-stp)):
      stpf = stpc
    else:
      stpf = stpq
    brackt = True
  elif (abs(dp) < abs(dx)):
    theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
    s = max(abs(theta), abs(dx), abs(dp))
    ths = theta/s
    gamma = s*np.sqrt(max(0.,ths*ths-(dx/s)*(dp/s)))
    if (stp > stx): gamma = -gamma
    p = (gamma - dp) + theta
    q = (gamma + (dx - dp)) + gamma
    r = p/q
    if (r < 0.0 and gamma != 0.0):
      stpc = stp + r*(stx - stp)
    elif (stp > stx):
      stpc = stpmax
    else:
      stpc = stpmin
   
    stpq = stp + (dp/(dp - dx))*(stx - stp)
   
    if (brackt):
      if (abs(stpc-stp) < abs(stpq-stp)):
        stpf = stpc
      else:
        stpf = stpq
      if (stp > stx):
        stpf = min(stp+0.66*(sty-stp),stpf)
      else:
        stpf = max(stp+0.66*(sty-stp),stpf)
    else:
      if (abs(stpc-stp) > abs(stpq-stp)):
        stpf = stpc
      else:
        stpf = stpq
      stpf = min(stpmax,stpf)
      stpf = max(stpmin,stpf)
  else:
    if (brackt):
      theta = 3.0*(fp - fy)/(sty - stp) + dy + dp
      s = max(abs(theta), abs(dy),abs(dp))
      ths = theta/s
      gamma = s*np.sqrt(ths*ths - (dy/s)*(dp/s))
      if (stp > sty): gamma = -gamma
      p = (gamma - dp) + theta
      q = ((gamma - dp) + gamma) + dy
      r = p/q
      stpc = stp + r*(sty - stp)
      stpf = stpc
    elif (stp > stx):
      stpf = stpmax
    else:
      stpf = stpmin

  if (fp > fx):
    sty = stp
    fy = fp
    dy = dp
  else:
    if (sgnd < 0):
      sty = stx
      fy = fx
      dy = dx;
    stx = stp
    fx = fp
    dx = dp
  stp = stpf

  return stx,fx,dx,sty,fy,dy,stp,brackt

def dcsrch(f: float, 
           g: float, 
           stp: float, 
           c1: float, 
           c2: float, 
           xtol: float, 
           stpmin: float, 
           stpmax: float,
           fg, 
           verbose: int = 0,
           grad = None,
           np=numpy):

  stpmin = min(stpmin, stpmax)

  xtrapl = 1.1
  xtrapu = 4.0

  if (g >= 0.0):
    if (verbose >= 99):
      print("ERROR: INITIAL G >= 0.0")
    return f, grad, stp, 'ERROR', 0


#        Initialize local variables.
  brackt = False
  stage = 1
  finit = f
  ginit = g
  gtest = c1*ginit
  width = stpmax - stpmin
  width1 = width/0.5
  '''
        The variables stx, fx, gx contain the values of the step,
        function, and derivative at the best step.
        The variables sty, fy, gy contain the value of the step,
        function, and derivative at sty.
        The variables stp, f, g contain the values of the step,
        function, and derivative at stp.
  '''
  stx = 0.0
  fx = finit
  gx = ginit
  sty = 0.0
  fy = finit
  gy = ginit
  stmin = 0.0
  stmax = stp + xtrapu*stp
  fg_cnt = 0


  for _ in range(100):
    f, g, grad = fg(stp)
    fg_cnt += 1
    stp_old = stp

    # finite
    for _i in range(20):
      if np.isneginf(f):
        break

      if np.isfinite(f) and np.isfinite(g).all():
        break

      if verbose >= 99:
        print('f or g has inf or nan')

      stp = 0.5 * stp
      stpmax = stp
      f, g, grad = fg(stp)
      fg_cnt += 1
    else:
      if verbose >= 99:
        print('Line search couldn\'t find a step inside feasible region')
      return f, grad, None, 'WARNING', fg_cnt   
    
    '''
     If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
     algorithm enters the second stage.
    '''
    ftest = finit + stp*gtest
    if (stage == 1 and f <= ftest and g >= 0.0):
      stage = 2

    # Test for warnings.
    task = ''
    if (brackt and (stp <= stmin or stp >= stmax)):
      task = 'WARNING'
      if (verbose >= 99):
          print("ROUNDING ERRORS PREVENT PROGRESS")
    if (brackt and stmax - stmin <= xtol*stmax):
      task = 'WARNING'
      if (verbose >= 99):
        print("XTOL TEST SATISFIED")
    if (stp == stpmax and f <= ftest and g <= gtest):
      task = 'CONVERGENCE'
      if (verbose >= 99):
        print("STP = STPMAX")
    if (stp == stpmin and (f > ftest or g >= gtest)):
      task = 'WARNING'
      if (verbose >= 99):
          print("STP = STPMIN")

    #     Test for convergence.
    eps: float = 0#1E-6
    if (f < ftest + eps*(abs(ftest) + 1) and abs(g) <= c2 * (-ginit)):
      task = 'CONVERGENCE'
      if verbose >= 99:
          print('Strong Wolfe satisfied')

    #     Test for termination.
    if (task == 'WARNING' or task == 'CONVERGENCE'):
      return f, grad, stp, task, fg_cnt

    if (stage == 1 and f <= fx and f > ftest):
      '''
       A modified function is used to predict the step during the
       first stage if a lower function value has been obtained but
       the decrease is not sufficient.
      '''
      #        Define the modified function and derivative values.
      fm = f - stp*gtest
      fxm = fx - stx*gtest
      fym = fy - sty*gtest
      gm = g - gtest
      gxm = gx - gtest
      gym = gy - gtest
      stx,fxm,gxm,sty,fym,gym,stp,brackt = dcstep(stx,fxm,gxm,sty,fym,gym,stp,fm,gm,brackt,stmin,stmax,verbose,np=np)

      #Reset the function and derivative values for f.
      fx = fxm + stx*gtest
      fy = fym + sty*gtest
      gx = gxm + gtest
      gy = gym + gtest
    else:
      stx,fx,gx,sty,fy,gy,stp,brackt = dcstep(stx,fx,gx,sty,fy,gy,stp,f,g,brackt,stmin,stmax,verbose,np=np)

    #     Decide if a bisection step is needed.
    if (brackt):
      if (abs(sty-stx) >= 0.66*width1): stp = stx + 0.5*(sty - stx)
      width1 = width
      width = abs(sty-stx)

    #     Set the minimum and maximum steps allowed for stp.
    if (brackt):
      stmin = min(stx,sty)
      stmax = max(stx,sty)
    else:
      stmin = stp + xtrapl*(stp - stx)
      stmax = stp + xtrapu*(stp - stx)
  
    # Force the step to be within the bounds stpmax and stpmin.
    stp = np.clip(stp, stpmin, stpmax)

    '''
       If further progress is not possible, let stp be the best
       point obtained during the search.
    '''
    if ((brackt and (stp <= stmin or stp >= stmax)) or (brackt and stmax-stmin <= xtol*stmax)):
      stp = stx


  # 20 iterations without CONVERGE
  if (verbose >= 99):
    print('MAX ITER reached')
  return f, grad, stp_old, 'WARNING', fg_cnt


def line_search_wolfe3(fg, xk, d, g=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50., amin=1e-14,
                       xtol=1e-14, verbose=0, np=numpy):

    stp = np.clip(1., amin, amax)
    
    def phi(s):
        fx, gx = fg(xk + s*d)
        gd = np.dot(gx, d)
        return fx, gd, gx
    
    cnt_on = 0
    if old_fval is None or g is None:
        old_fval, _, g = phi(0)
        cnt_on = 1

    fval = old_fval
    gd = np.dot(g, d)
    fval, g, stp, ret, fg_cnt = dcsrch(fval, gd, stp, c1, c2, xtol,
                                        amin, amax, phi, verbose, 
                                        grad=g, np=np)

    if old_fval is None or g is None:
        fg_cnt += 1
    
    if ret == 'ERROR':
        print('Error in line search')
        raise Exception('Line search error')

    return stp, (fg_cnt+cnt_on), fval, g
















def line_search_wolfe3_debug(fg, xk, pk, gfk=None,
                             old_fval=None, old_old_fval=None,
                             args=(), c1=1e-4, c2=0.9, amax=50., amin=1e-14,
                             xtol=0.1, verbose=100, np=numpy, plot_path=None):

    stp = 1.
    stp = max(amin, stp)
    stp = min(stp, amax)
    
    ftol = c1
    gtol = c2

    steps_array = []
    def phi(s):
        steps_array.append(s)
        fx, gx = fg(xk + s*pk)
        gd = np.dot(gx, pk)
        return fx, gd, gx
    
    cnt_on = 0
    if old_fval is None or gfk is None:
        old_fval, _, gfk = phi(0)
        cnt_on = 1

    fval = old_fval
    gval = np.dot(gfk, pk)
    fval, grad, stp, task, fg_cnt = dcsrch(fval, gval, stp, ftol, gtol, xtol, amin, amax, task, phi, verbose, grad=gfk, np=np)

    if old_fval is None or gfk is None:
        fg_cnt += 1
    if stp:
        steps_array.append(stp)
    if stp != 1.:
        import matplotlib.pyplot as plt
        import os, datetime
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        step_space = np.linspace(0., max(steps_array), 50)
        axs[0].plot(step_space, [ fg(xk + s*pk)[0] for s in step_space ])
        axs[0].plot(steps_array, [ fg(xk + s*pk)[0] for s in steps_array], '.')
        axs[1].semilogy(list(range(len(steps_array))), steps_array)
        if not plot_path is None: 
          os.makedirs(plot_path, exist_ok=True)
          plot_name = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
          plt.save(os.path.join(plot_path, f'{plot_name}.pdf'))
        else:
            plt.show()
        plt.close()
    print(task)
    if task == 'ERROR':
        print('Error in line search')
        raise Exception('Line search error')

    return stp, (fg_cnt+cnt_on), 0, fval, old_fval, grad
