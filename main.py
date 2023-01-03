from regretmatching.blotto import BLOTTOPlayer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

blotto_strategies = [(5,0,0), (4,0,1), (4,1,0), (3,2,0), (3,0,2), (3,1,1), (2,3,0), (2,0,3), (2,2,1), 
(2,1,2), (1,4,0), (1,0,4), (1,1,3), (1,3,1), (1,2,2), (0,5,0), (0,4,1), (0,3,2), (0,2,3), (0,1,4), (0,0,5)]
a = BLOTTOPlayer()
b = BLOTTOPlayer()
t = 3000001
alist = []
blist = []
ylist = []
for i in range(0, t):
    a_move = a.move()
    b_move = b.move()
    a.learn_from(b_move)
    b.learn_from(a_move)
    alist.append(a.current_best_response())
    blist.append(b.current_best_response())
    ylist.append(i)
   


_2e = np.round(2 * np.max([a.eps(), b.eps()]), 10)
a_ne = a.current_best_response()
b_ne = b.current_best_response()
plt.plot(ylist, alist, color='blue', alpha=0.3)
plt.plot(ylist, blist, color='red', alpha=0.3)
plt.title('Graph showing how the propabilities of the strategies of Colonel Blotto and Boba Fett converge')
plt.ylabel('Probability strategy is optimal')
plt.xlabel('Time (Number of Iterations)')
blue_line = mlines.Line2D([], [], color='blue',
                          markersize=15, label='Colonel Blotto')
red_line = mlines.Line2D([], [], color='red',
                          markersize=15, label='Boba Fett')
plt.legend(handles=[blue_line, red_line])
plt.show()
print("{0} - nash equilibrium for BLOTTO: {1}, {2}".format(_2e, a_ne, b_ne))
u1= a.utility()
print(str(u1))
