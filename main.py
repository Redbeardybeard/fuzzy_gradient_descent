import math
import matplotlib.pyplot as plt


def calc_u_k(step):
    incoming = []
    for every_unit in step:
        incoming.append(math.sin((math.pi*every_unit)/100))
    return incoming


def calc_g_u_k(inp):
    g_out = []
    for every_unit in inp:
        g_out.append(0.6*math.sin(math.pi*every_unit)+0.3*math.sin(math.pi*3*every_unit)+0.1*math.sin(math.pi*5*every_unit))
    return g_out


def make_f(xbar, ybar, sigma, inp):  # M must be declared before calling this method
    ai = 0
    bi = 0
    count = 0
    while count < M:
        ai += ybar[count]*math.exp(-math.pow((inp-xbar[count])/sigma[count], 2))
        bi += math.exp(-math.pow((inp-xbar[count])/sigma[count], 2))
        count += 1
    fi = ai/bi
    return fi


def calc_error(foutp, goutp):
    er = 0.5*math.pow(foutp-goutp, 2)
    return er


def gradient_descent(current_rule, inp, alpha, xbar, ybar, sigma, foutp, goutp):
    resulting_params = []  # 1. xbar 2.ybar 3.sigma
    ZL = math.exp(-math.pow((inp-xbar[current_rule])/sigma[current_rule], 2))
    B = 0
    count = 0
    while count < M:
        B += math.exp(-math.pow((inp-xbar[count])/sigma[count], 2))
        count += 1
    new_ybar = ybar[current_rule] - alpha*(foutp-goutp)*(ZL/B) # new ybar is calculated
    print(ybar[current_rule]," was our ybar,",new_ybar," this is our new ybar.and it moved for:", ybar[current_rule]-new_ybar)

    new_xbar = xbar[current_rule]- alpha*(foutp-goutp)*2*(ZL/B)*(ybar[current_rule]-foutp)*((inp-xbar[current_rule])/math.pow(sigma[current_rule],2))
    print(xbar[current_rule]," was our xbar,",new_xbar," this is our new xbar.and it moved for:", xbar[current_rule]-new_xbar)

    new_sig = sigma[current_rule] - alpha*(foutp-goutp)*2*(ZL/B)*(ybar[current_rule]-foutp)*math.pow((inp-xbar[current_rule]),2)*math.pow(1/(sigma[current_rule]),3)
    print(sigma[current_rule]," was our sigma,",new_sig," this is our new sigma.and it moved for:", sigma[current_rule]-new_sig)

    resulting_params.append(new_xbar)
    resulting_params.append(new_ybar)
    resulting_params.append(new_sig)
    return resulting_params


def calc_real_outputs(rout, out):
    i = 2
    for x in out:
        rout.append(0.3*rout[i-1]+0.6*rout[i-2]+x)
        i += 1
    return rout


M = 10
POINT = [*range(0, 500, 1)]
IN = calc_u_k(POINT)
GOUT = calc_g_u_k(IN)
FOUT = []

XBAR = [IN[M-1]]*M
YBAR = [GOUT[M-1]]*M
SIGMA = [(max(IN[0:M])-min(IN[0:M]))/M]*M

RULE = 0
STEP = 0
pts = 0  # points fed to system for current rule
status = True
loop_counter = 0
nott = 0  # number of times trained

STEP = M-1
print("building f with initial values...")
F = make_f(XBAR, YBAR, SIGMA, IN[STEP])
print("F is : ", F, " at point: ", STEP, " with input: ", IN[STEP])
print("G is : ", GOUT[STEP], " at point: ", STEP, " with input: ", IN[STEP])

while status:
    if STEP >= 211:
        print("training ended")
        print("FINAL DETAILS: <------------------------------------------>")
        print("xbar : ", XBAR)
        print("ybar : ", YBAR)
        print("sigma : ", SIGMA)
        decision = input("continue with drawing with the approximated function?(y/n): ")
        if decision == "y":
            while STEP != 500:
                F = make_f(XBAR, YBAR, SIGMA, IN[STEP])
                FOUT.append(F)
                STEP += 1
        break
    loop_counter += 1
    print("counter is at:", loop_counter,"\n########################")
    ERR = calc_error(F, GOUT[STEP])
    print("error rate is: ", ERR)
    if ERR <= 0.001:  # if error rate is less than epsilon(0.01) in this case we feed next point to the system.
        FOUT.append(F)
        print("good accuracy, next step at: ",STEP+1)
        STEP += 1
        F = make_f(XBAR, YBAR, SIGMA, IN[STEP])
        continue
    else:
        print("training...")
        level = 0
        while True:
            print("training rule : ",level,"with input: ",IN[STEP],"current info:\n"
                                                                   "xbar=",XBAR,"\nybar=",YBAR,"\nsigma=",SIGMA)
            change = gradient_descent(level, IN[STEP], 0.5, XBAR, YBAR, SIGMA, F, GOUT[STEP])
            XBAR[level] = change[0]
            YBAR[level] = change[1]
            SIGMA[level] = change[2]
            F = make_f(XBAR, YBAR, SIGMA, IN[STEP])
            ERR = calc_error(F, GOUT[STEP])
            if ERR > 0.001 and nott < 10:
                level += 1
                if level == M:
                    nott += 1
                    level = 0
                continue
            if ERR > 0.001 and nott == 10:
                nott = 0
                print("cant further optimize at step: ", STEP)
                STEP += 1
                F = make_f(XBAR, YBAR, SIGMA, IN[STEP])
                break
            if ERR < 0.001:
                FOUT.append(F)
                STEP +=1
                break


# check ranges at the end
yout = [0,0]
fyout = [0,0]  # debatable
fullfyout=[0,0]

yout = calc_real_outputs(yout,GOUT)
fyout = calc_real_outputs(fyout,FOUT)

#plt.figure(1)
#plt.plot(POINT, yout[2:])
#plt.plot(POINT[10:499], fyout)

#plt.figure(2)
#plt.plot(POINT, GOUT)
#plt.plot(POINT[10:497], FOUT)

plt.figure(3)
plt.plot(POINT, yout[2:])
plt.plot(POINT[10:210], fyout)

plt.figure(4)
plt.plot(POINT, GOUT)
plt.plot(POINT[10:208], FOUT)


plt.show()

