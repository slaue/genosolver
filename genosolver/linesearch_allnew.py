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
import numpy as np

def dcstep(stx: float, # &
           fx: float, # &
           dx: float, # &
           sty: float, # &
           fy: float, # &
           dy: float, # &
           stp: float, # &
           fp: float,
           dp: float,
           brackt: bool, # &
           stpmin: float, # const
           stpmax: float, # const
           verbose: int):
  '''
  double const zero  = 0.0,
               p66   = 0.66,
               two   = 2.0,
               three = 3.0;
  double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta;
  '''
  sgnd = dp * np.sign(dx)
  if (fp > fx):
    #    std::cout << "dcstep 1" << std::endl;
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
    #    std::cout << "dcstep 2" << std::endl;
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
    #    std::cout << "dcstep 3 ***************" << std::endl;
    theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
    s = max(abs(theta), abs(dx), abs(dp))
    ths = theta/s
    gamma = s*np.sqrt(max(0.,ths*ths-(dx/s)*(dp/s)))
    #     std::cout << "theta = " << theta << std::endl;
    #     std::cout << "gamma = " << gamma << std::endl;
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
    #     std::cout << "stpc = " << stpc << std::endl;

    stpq = stp + (dp/(dp - dx))*(stx - stp)
    '''
    //     std::cout << "--------------------------------------------" << std::endl;
    //     std::cout << "stpq = " << stpq << std::endl;
    //     std::cout << "stp  = " << stp << std::endl;
    //     std::cout << "dp   = " << dp << std::endl;
    //     std::cout << "dx   = " << dx << std::endl;
    //     std::cout << "stx  = " << stx << std::endl;
    //     std::cout << "dp/(dp-dx) = " << dp/(dp-dx) << std::endl;
    //     std::cout << "--------------------------------------------" << std::endl;
    '''


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
    #    std::cout << "dcstep 4" << std::endl;
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

def dcsrch(f: float, # &
           g: float, # &
           stp: float, # &
           ftol: float, # const
           gtol: float, # const
           xtol: float, # const
           stpmin: float, # const, should always set to zero
           stpmax: float, # const
           task: str, # TaskType&
           fg, # !!!!!!!
           verbose: int = 0,
           grad = None):
  '''
  double const zero = 0.0,
               p5   = 0.5,
               p66  = 0.66,
               xtrapl = 1.1,
               xtrapu = 4.0;
  static bool brackt;
  static int stage;
  static double finit, fx, fy, ginit,gtest, gx, gy,
         stx,sty,stmin,stmax,width,width1;
  double ftest,fm,fxm,fym, gm,gxm,gym;
  '''
  xtrapl = 1.1
  xtrapu = 4.0

  #if (task == 'START'):
  #       Check the input arguments for errors.
  if (stp < stpmin):
    task = 'ERROR'
    if (verbose >= 99):
      print("ERROR: STP < STPMIN")
  if (stp > stpmax):
    task = 'ERROR'
    if (verbose >= 99):
      print("ERROR: STP > STPMAX")
  if (g >= 0.0):
    task = 'ERROR'
    if (verbose >= 99):
      print("ERROR: INITIAL G >= 0.0")
  if (ftol < 0.0):
    task = 'ERROR'
    if (verbose >= 99):
      print("ERROR: FTOL < 0.0")
  if (gtol < 0.0):
    task = 'ERROR'
    if (verbose >= 99):
      print("ERROR: GTOL < 0.0")
  if (xtol < 0.0):
    task = 'ERROR'
    if (verbose >= 99):
      print("ERROR: XTOL < 0.0")
  if (stpmin < 0.0):
    task = 'ERROR'
    if (verbose >= 99):
      print("ERROR: STPMIN < 0.0")
  if (stpmax < stpmin):
    task = 'ERROR'
    if (verbose >= 99):
      print("ERROR: STPMAX < STPMIN")

        #        Exit if there are errors on input.
  if (task == 'ERROR'):
    return f, grad, stp, task, 0

#        Initialize local variables.

  brackt = False
  stage = 1
  finit = f
  ginit = g
  gtest = ftol*ginit
  width = stpmax - stpmin
  width1 = width/0.5
  '''
//        The variables stx, fx, gx contain the values of the step,
//        function, and derivative at the best step.
//        The variables sty, fy, gy contain the value of the step,
//        function, and derivative at sty.
//        The variables stp, f, g contain the values of the step,
//        function, and derivative at stp.
  '''
  stx = 0.0
  fx = finit
  gx = ginit
  sty = 0.0
  fy = finit
  gy = ginit
  stmin = 0.0
  stmax = stp + xtrapu*stp
  #task = 'FG'
  fg_cnt = 0

  #return f, g, stp, task
  # end taks == 'START'

  '''
//     If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
//     algorithm enters the second stage.
  '''
  for _ in range(20):
    f, g, grad = fg(stp)
    fg_cnt += 1
    stp_old = stp

    # finite
    for _i in range(10):
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
    
    ftest = finit + stp*gtest
    if (stage == 1 and f <= ftest and g >= 0.0):
      stage = 2

    # Test for warnings.

    if (brackt and (stp <= stmin or stp >= stmax)):
      task = 'WARNING'
      if (verbose >= 99):
          print("ROUNDING ERRORS PREVENT PROGRESS")
    if (brackt and stmax - stmin <= xtol*stmax):
      task = 'WARNING'
      if (verbose >= 99):
        print("XTOL TEST SATISFIED")
    if (stp == stpmax and f <= ftest and g <= gtest):
        #//    task = TaskType::WARNING;
      task = 'CONVERGENCE'
      if (verbose >= 99):
        print("STP = STPMAX")
    if (stp == stpmin and (f > ftest or g >= gtest)):
      task = 'WARNING'
      if (verbose >= 99):
          print("STP = STPMIN")
    #     Test for convergence.

    '''
          printf("f = %15.15g \n", f);
          printf("ftest = %15.15g\n", ftest);
          std::cout << "f " << f << std::endl;
          std::cout << "ftest " << ftest << std::endl;
          std::cout << "fabs(g) " << std::fabs(g) << std::endl;
          std::cout << "gtol*ginit " << gtol * (-ginit) << std::endl;
    '''
    #      if (f <= ftest && std::fabs(g) <= gtol * (-ginit))
    eps: float = 0#1E-6
    if (f < ftest + eps*(abs(ftest) + 1) and abs(g) <= gtol * (-ginit)):
      task = 'CONVERGENCE'
      if verbose >= 99:
          print('Strong Wolfe satisfied')

    #     Test for termination.

    if (task == 'WARNING' or task == 'CONVERGENCE'):
      return f, grad, stp, task, fg_cnt
    '''
  //     A modified function is used to predict the step during the
  //     first stage if a lower function value has been obtained but
  //     the decrease is not sufficient.
    '''
    if (stage == 1 and f <= fx and f > ftest):
      #        Define the modified function and derivative values.
      fm = f - stp*gtest
      fxm = fx - stx*gtest
      fym = fy - sty*gtest
      gm = g - gtest
      gxm = gx - gtest
      gym = gy - gtest
      #Call dcstep to update stx, sty, and to compute the new step.
      #std::cout << "called dcstep 1" << std::endl;
      stx,fxm,gxm,sty,fym,gym,stp,brackt = dcstep(stx,fxm,gxm,sty,fym,gym,stp,fm,gm,brackt,stmin,stmax, verbose)
      #Reset the function and derivative values for f.
      fx = fxm + stx*gtest
      fy = fym + sty*gtest
      gx = gxm + gtest
      gy = gym + gtest
    else:
      #Call dcstep to update stx, sty, and to compute the new step.
      #std::cout << "called dcstep 2" << std::endl;
      stx,fx,gx,sty,fy,gy,stp,brackt = dcstep(stx,fx,gx,sty,fy,gy,stp,f,g,brackt,stmin,stmax, verbose)
      '''
      // std::cout << "stx = " << stx << std::endl;
      // std::cout << "fx  = " << fx << std::endl;
      // std::cout << "gx  = " << gx << std::endl;
      // std::cout << "sty = " << sty << std::endl;
      // std::cout << "fy  = " << fy << std::endl;
      // std::cout << "gy  = " << gy << std::endl;
      // std::cout << "stp = " << stp << std::endl;
      // std::cout << "f   = " << f << std::endl;
      // std::cout << "g   = " << g << std::endl;
      // std::cout << "stmin = " << stmin << std::endl;
      // std::cout << "stmax = " << stmax << std::endl;
      '''
    #     Decide if a bisection step is needed.

    if (brackt):
      # std::cout << "brackt true" << std::endl;
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
    '''
  //
  //     Force the step to be within the bounds stpmax and stpmin.
  //
    '''
    stp = max(stp,stpmin)
    stp = min(stp,stpmax)

    '''
  //     If further progress is not possible, let stp be the best
  //     point obtained during the search.
    '''
    if ((brackt and (stp <= stmin or stp >= stmax)) or (brackt and stmax-stmin <= xtol*stmax)):
      # std::cout << "no further progress" << std::endl;
      stp = stx

    #Obtain another function and derivative.

    #task = 'FG'
    #return f, g, stp, task
    # continue loop of FG

  # 20 iterations without CONVERGE
  task = 'WARNING'
  if (verbose >= 99):
    print('MAX ITER')
  return f, grad, stp_old, task, fg_cnt
