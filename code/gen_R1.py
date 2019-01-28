import numpy as np
# coding: utf-8

def build_R1_for_rot(ang, cnt, n):
    #print cnt
    tot_dim = sum(cnt)
    R = np.zeros((tot_dim, tot_dim))
    o = 1.
    cur_r = cnt[0]
    R[:cnt[0], :cnt[0]] = np.eye(cnt[0])
    for c in cnt[1:]:
        r = np.zeros((2, 2))
        cc = np.cos(o * np.deg2rad(ang))
        ss = np.sin(o * np.deg2rad(ang))
        r[0, 0] = r[1, 1] = cc
        r[0, 1] = -ss
        r[1, 0] = ss
        for i in range(c/2):
            R[cur_r:cur_r+2, cur_r:cur_r+2] = r
            cur_r += 2
        o += 1
    return R

def build_R2_for_rot(ang, cnt, n):
    print cnt
    tot_dim = sum(cnt)
    R = np.zeros((tot_dim, tot_dim))
    o = 1.
    cur_r = cnt[0]
    R[:cnt[0], :cnt[0]] = np.eye(cnt[0])
    for c in cnt[1:]:
        r = np.zeros((2, 2))
        cc = np.cos(np.deg2rad(ang) / o)
        ss = np.sin(np.deg2rad(ang) / o)
        r[0, 0] = r[1, 1] = cc
        r[0, 1] = ss
        r[1, 0] = ss
        for i in range(c/2):
            R[cur_r:cur_r+2, cur_r:cur_r+2] = r
            cur_r += 2
        o += 1
    return R

def build_R1_for_rot_tf(ang, cnt, n):
    print cnt
    tot_dim = sum(cnt)
    R = np.zeros((tot_dim, tot_dim))
    o = 1.
    cur_r = cnt[0]
    R[:cnt[0], :cnt[0]] = np.eye(cnt[0])
    for c in cnt[1:]:
        r = np.zeros((2, 2))
        cc = np.cos(o * (ang))
        ss = np.sin(o * (ang))
        r[0, 0] = r[1, 1] = cc
        r[0, 1] = -ss
        r[1, 0] = ss
        for i in range(c/2):
            R[cur_r:cur_r+2, cur_r:cur_r+2] = r
            cur_r += 2
        o += 1
    return R
