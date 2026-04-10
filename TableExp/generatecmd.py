import math
import os
import shutil
from hashlib import algorithms_available
import numpy as np


gammalist = [0.95]
epsilonlist = [1]
epsilonexp = 0.5
lrlist = [1]
lrexplist = [0.8]

adplr = "True"
adpepsilon = "True"
use_episode = "False"
episodes = 10000

steps = 10000
total_repeat = 1000
simpleinfo = "no"


gridrlist_orginal = [

            #(-6,4,-30, 40),
            (-3, 2,-5,10),
            (-6,4,-10, 20),
            (-12,8,-20, 40),
            (-24,16,-40, 80),


             ]
nrowlist = [
    (3,3),
    #(4, 12),
    #(4,4),
    #(5,5),
    # 7,
    ]
rightEnvMeanVar_list_orginal = [
    (-0.1, 0, 1, 1),
    # (-1, 0, 1,1),
    # (-0.5,2 ,1, 2),
]

envlist = [
    #"rightenv",
    #"grid",
    #"ABCenv",
    "multiarmenv",
]

Klist = [2]

algorithmlist = [
    "AveragedQ",
    # "MaxminQ",
    # "WeightedQ",
    "EBQL",
    "DoubleQ",
    "SingleQ",
    # "AdaOQ",
    #"EnsembleQ",
    #"AMultiplexQ",
    "rhoFixOverQ",
    #"mixDQ",
    #"ACCDQ"
]

# if 2 in Klist:
#     algorithmlist.append("AMultiplexQ")
# if 16 in Klist:
#     algorithmlist.append("rhoFixOverQ")


multiarmcfgs_orignal = []

Narms = [5,10,15,20]
armstds = [5, 10, 15, 20]
Qstds = [1, 2, 4, 8]

Narms = [10,10,10,10]
armstds = [10, 10, 10, 10]
Qstds = [0.01, 0.01, 0.01, 0.01]

defaultNarm = 10
defaultarmstd = 10
defaultQmean = 0
defaultQstd = 0.01



for Narm in Narms:
    armstd = defaultarmstd
    Qmean = defaultQmean
    Qstd = defaultQstd
    tmpcfg = (Narm, armstd, Qmean, Qstd)
    if tmpcfg not in multiarmcfgs_orignal:
        multiarmcfgs_orignal.append(tmpcfg)

for armstd in armstds:
    Narm = defaultNarm
    Qmean = defaultQmean
    Qstd = defaultQstd
    tmpcfg = (Narm, armstd, Qmean, Qstd)
    if tmpcfg not in multiarmcfgs_orignal:
        multiarmcfgs_orignal.append(tmpcfg)

for Qstd in Qstds:
    Narm = defaultNarm
    armstd = defaultarmstd
    Qmean = defaultQmean
    tmpcfg = (Narm, armstd, Qmean, Qstd)
    if tmpcfg not in multiarmcfgs_orignal:
        multiarmcfgs_orignal.append(tmpcfg)

Qinitmode = "ALLDIFF"



if __name__ == "__main__":
    for env in envlist:
        if env in ["rightenv", "twoenv", "ABCenv", "Trightenv10", "Trightenv2", "Roulette"]:
            optimalaction = "True"
            rightEnvMeanVar_list = rightEnvMeanVar_list_orginal
            gridrlist = [(-12,8, -20, 40)]
            multiarmcfgs=[(1,1,0,0.01)]

        elif env in ["grid"]:
            optimalaction = "False"
            rightEnvMeanVar_list = [(-0.1, 0, 1, 1)]
            gridrlist = gridrlist_orginal
            multiarmcfgs = [(1, 1, 0, 0.01)]
        elif env in ["multiarmenv"]:
            optimalaction = "True"
            rightEnvMeanVar_list = [(-0.1, 0, 1, 1)]
            gridrlist = [(-12,8, -20, 40)]
            multiarmcfgs=multiarmcfgs_orignal
        else:
            raise NotImplementedError

        if "rhoFixOverQ" in algorithmlist:
            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                for lrexp in lrexplist:
                    for k in [16]:
                        mlist = np.arange(int(k/2), k+1)
                        for m in mlist:
                            for lr in lrlist:
                                for gamma in gammalist:
                                    for epsilon in epsilonlist:
                                        for r1, r2, gr1, gr2 in gridrlist:
                                            for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                print(f"nohup python QMain.py --env {env} --algorithm rhoFixOverQ --K {k} --M {m} --gamma {gamma} --epsilon {epsilon} --lr {lr} --adplr {adplr} --adpepsilon {adpepsilon} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction} --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > rhoFixK{k}.out 2>&1 &")


        #SingleQ
        if "SingleQ" in algorithmlist:

            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for nrow, ncol in nrowlist:
                                        for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                            print(f"nohup python QMain.py --env {env} --algorithm SingleQ --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction} --adplr {adplr} --adpepsilon {adpepsilon} --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > SingleQ.out 2>&1 &")

        # DoubleQ
        if "DoubleQ" in algorithmlist:
            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for nrow, ncol in nrowlist:
                                        for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                            print(f"nohup python QMain.py --env {env} --algorithm DoubleQ --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction} --adplr {adplr} --adpepsilon {adpepsilon} --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > DoubleQ.out 2>&1 &")


        if "mixDQ" in algorithmlist:
            mixratiolist = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                            0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            #mixratiolist = [0.26, 0.27, 0.28, 0.29]
            mixratiolist = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            #mixratiolist = [1]
            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for mixratio in mixratiolist:
                                        for nrow, ncol in nrowlist:
                                            for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                print(
                                                f"nohup python QMain.py --env {env} --algorithm mixDQ --mixratio {mixratio} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction} --adplr {adplr} --adpepsilon {adpepsilon} --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > mixDQ.out 2>&1 &")

        if "AMultiplexQ" in algorithmlist:
            mixratiolist = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                            0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            #mixratiolist = [0.26, 0.27, 0.28, 0.29]
            mixratiolist = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            mixratiolist = [1]
            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for mixratio in mixratiolist:
                                        topKlist = [2,3,4,5,6,7,8]
                                        for topK in topKlist:
                                            for nrow, ncol in nrowlist:
                                                for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                    print(
                                                    f"nohup python QMain.py --env {env} --algorithm AMultiplexQ --multiplexratio {mixratio} --K {topK} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction} --adplr {adplr} --adpepsilon {adpepsilon} --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > AMultiplexQ.out 2>&1 &")
        if "ACCDQ" in algorithmlist:
            mixratiolist = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                            0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            #mixratiolist = [0.26, 0.27, 0.28, 0.29]
            mixratiolist = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            mixratiolist = [1]
            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for mixratio in mixratiolist:
                                        topKlist = [2,3,4,5,6,7,8]
                                        for topK in topKlist:
                                            for nrow, ncol in nrowlist:
                                                for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                    print(
                                                    f"nohup python QMain.py --env {env} --algorithm ACCDQ --multiplexratio {mixratio} --K {topK} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction} --adplr {adplr} --adpepsilon {adpepsilon} --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > ACCDQ.out 2>&1 &")



        if "AveragedQ" in algorithmlist:

            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for K in Klist:
                                        for nrow, ncol in nrowlist:
                                            for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                print(
                                                f"nohup python QMain.py --env {env} --algorithm AveragedQ --K {K} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction}  --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > AveragedQK{K}.out 2>&1 &")

        if "EnsembleQ" in algorithmlist:

            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for K in Klist:
                                        for nrow, ncol in nrowlist:
                                            for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                print(
                                                f"nohup python QMain.py --env {env} --algorithm EnsembleQ --K {K} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction}  --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > EnsembleQK{K}.out 2>&1 &")

        if "MaxminQ" in algorithmlist:

            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for nrow, ncol in nrowlist:
                                        for K in Klist:
                                            for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                print(f"nohup python QMain.py --env {env} --algorithm MaxminQ --K {K} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction}  --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > MaxminQ.out 2>&1 &")

        if "WeightedQ" in algorithmlist:
            clist = [1, 10]
            adpc = "True"
            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for c in clist:
                                        for nrow, ncol in nrowlist:
                                            for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                print(
                                                f"nohup python QMain.py --env {env} --algorithm WeightedQ --c {c} --adpc {adpc} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction}  --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > WeightedQc{c}.out 2>&1 &")


        if "EBQL" in algorithmlist:

            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for K in Klist:
                                        for nrow, ncol in nrowlist:
                                            for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                print(
                                                f"nohup python QMain.py --env {env} --algorithm EBQL --K {K} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction}  --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > EBQLK{K}.out 2>&1 &")

        if "KQ" in algorithmlist:

            for lr in lrlist:
                for lrexp in lrexplist:
                    for gamma in gammalist:
                        for epsilon in epsilonlist:
                            for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                for r1, r2, gr1, gr2 in gridrlist:
                                    for K in Klist:
                                        for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                            print(f"nohup python QMain.py --env {env} --algorithm KQ --K {K} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction} --adplr {adplr} --adpepsilon {adpepsilon} --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > KQ.out 2>&1 &")

        if "AdaOQ" in algorithmlist:

            for K in Klist:
                Mm_list = [(K, int(K/2)), (K, K)]
                Mm_list = [(K,2)]
                for lr in lrlist:
                    for lrexp in lrexplist:
                        for gamma in gammalist:
                            for epsilon in epsilonlist:
                                for submean, optimalmean, subvar, optimalvar in rightEnvMeanVar_list:
                                    for r1, r2, gr1, gr2 in gridrlist:
                                        for nrow, ncol in nrowlist:
                                            for M,m in Mm_list:
                                                for Narm, armstd, Qmean, Qstd in multiarmcfgs:
                                                    print(f"nohup python QMain.py --env {env} --algorithm AdaOQ --M {M} --m {m} --gamma {gamma} --epsilon {epsilon} --lr {lr} --lrexp {lrexp} --epsilonexp {epsilonexp} --submean {submean} --optimalmean {optimalmean} --subvar {subvar} --optimalvar {optimalvar} --optimalaction {optimalaction} --adplr {adplr} --adpepsilon {adpepsilon} --gridr1 {r1} --gridr2 {r2} --gridgr1 {gr1} --gridgr2 {gr2} --nrow {nrow} --ncol {ncol} --steps {steps} --total_repeat {total_repeat} --Narm {Narm} --armstd {armstd} --Qmean {Qmean} --Qstd {Qstd} --info {simpleinfo} --Qinitmode {Qinitmode} > AdaOQ.out 2>&1 &")








