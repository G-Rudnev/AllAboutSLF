import math as m
from math import sin, cos, pi, atan2, tan
import numpy as np
import numpy.random as random
# from random import randint, random, uniform, normalvariate

import threading
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

N = 50
Nlines = 0
pnts = np.zeros([3, N])
pntsBuf = np.zeros([2, N])
# lines = np.zeros([4, N - 1]) ##################################################################################################
lines = np.zeros([2, N], dtype=float)  # хранит столбцы координат N-1 столбцов координат точек узлов ломаной

# sums = np.zeros([5], dtype=float)

tol = 0.1
mess = 0.1
shape = 0
mode = 'lsm'  # метод аппроксимации
timing = np.zeros(2, dtype=float)  # значения времени исполнения аппроксимации avg и lsm

fig = plt.figure()
ax = plt.axes([0.07, 0.25, 0.45, 0.7])

lidar_ax = (0.0, 0.0)

mutex = threading.RLock()

act_ostacles = None


def lineApproxAveraging(pnts: np.ndarray, fr: int, to: int):
    b0 = np.exp(np.linspace(-0.25, 1 - 0.25, (to - fr))**2 / -0.03)
    # b0 = np.linspace(1.0, 0.0, N)**2.0
    b0 /= np.sum(b0)

    b1 = np.flip(b0)

    pnt0 = [np.sum(b0 * pnts[0, fr : to]), np.sum(b0 * pnts[1, fr : to])]
    pnt1 = [np.sum(b1 * pnts[0, fr : to]), np.sum(b1 * pnts[1, fr : to])]

    a = (pnt1[1] - pnt0[1]) / (pnt1[0] - pnt0[0])
    b = pnt0[1] - a * pnt0[0]
    return (a, b, np.mean(((a * pnts[0, fr : to] - pnts[1, fr : to] + b)**2) / (b**2 + 1)))


# ====== РЕАЛИЗАЦИЯ ФУНКЦИЙ МНК С ИСПОЛЬЗОВАНИЕМ NUMPY =======

def calc_loss_func(seg_pnts: np.ndarray, A: float, C: float):
    loss_func_value = 0
    for i in range(seg_pnts.shape[1]):
        # Расстояние до текущей прямой
        dist = abs(seg_pnts[0, i] * A - seg_pnts[1, i] + C) / m.sqrt(A ** 2 + 1)
        loss_func_value += dist ** 2
    return loss_func_value


def calc_sums(seg_pnts: np.ndarray):
    x_sum = np.sum(seg_pnts[0])  # сумма всех х координат
    y_sum = np.sum(seg_pnts[1])  # сумма всех у координат
    xy_sum = np.dot(seg_pnts[0], seg_pnts[1].T)  # сумма произведений всех х и всех у координат соответственно
    x_sq_sum = np.dot(seg_pnts[0], seg_pnts[0].T)  # сумма квадратов всех координат х
    y_sq_sum = np.dot(seg_pnts[1], seg_pnts[1].T)  # сумма квадратов всех координат у

    return x_sum, y_sum, xy_sum, x_sq_sum, y_sq_sum


def line_approx_lsm(pnts: np.ndarray, fr: int, to: int):
    pts_num = to - fr
    x_sum, y_sum, xy_sum, x_sq_sum, y_sq_sum = calc_sums(pnts[:2, fr:to])
    # Вычисление A для минимумов функции потерь
    phi = xy_sum - x_sum * y_sum / pts_num
    if phi == 0:
        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        print(x_sum, y_sum, xy_sum, x_sq_sum, y_sq_sum, pts_num)
    theta = (x_sq_sum - y_sq_sum) / phi + (y_sum ** 2 - x_sum ** 2) / (pts_num * phi)
    D = theta ** 2 + 4  # дискриминант
    A1 = (-theta + m.sqrt(D)) / 2
    A2 = (-theta - m.sqrt(D)) / 2
    # Вычисление С для минимумов функции потерь
    C1 = (y_sum - x_sum * A1) / pts_num
    C2 = (y_sum - x_sum * A2) / pts_num
    # Подстановка в функцию потерь, выявление лучшего
    lf1 = calc_loss_func(pnts[:2, fr:to], A1, C1)
    lf2 = calc_loss_func(pnts[:2, fr:to], A2, C2)
    # Выбор наименьшего значения функции потерь, возврат соответствующих ему параметров А и С
    if lf1 < lf2:
        return A1, C1, np.mean(((A1 * pnts[0, fr:to] - pnts[1, fr:to] + C1) ** 2) / (A1 ** 2 + 1))
    else:
        return A2, C2, np.mean(((A2 * pnts[0, fr:to] - pnts[1, fr:to] + C2) ** 2) / (A2 ** 2 + 1))

# =====================================================================


def calc_sums_opt(seg_pnts: np.ndarray):
    # global sums
    sums = np.zeros([5], dtype=float)
    new_pt = seg_pnts[:2, -1]
    sums[0] = sums[0] + new_pt[0]
    sums[1] = sums[1] + new_pt[1]
    sums[2] = sums[2] + new_pt[0] * new_pt[1]
    sums[3] = sums[3] + new_pt[0] ** 2
    sums[4] = sums[4] + new_pt[1] ** 2
    return sums
    # print(sums)


# Попытка оптимизации LSM
def line_approx_opt_lsm(pnts: np.ndarray, fr: int, to: int):
    # global sums
    pts_num = to - fr
    sums = calc_sums_opt(pnts[:2, fr:to])

    phi = sums[2] - sums[0] * sums[1] / pts_num
    theta = (sums[3] - sums[4]) / phi + (sums[1] ** 2 - sums[0] ** 2) / (pts_num * phi)

    D = theta ** 2 + 4  # дискриминант
    A1 = (-theta + m.sqrt(D)) / 2
    A2 = (-theta - m.sqrt(D)) / 2
    # Вычисление С для минимумов функции потерь
    C1 = (sums[1] - sums[0] * A1) / pts_num
    C2 = (sums[1] - sums[0] * A2) / pts_num
    # Подстановка в функцию потерь, выявление лучшего
    lf1 = calc_loss_func(pnts[:2, fr:to], A1, C1)
    lf2 = calc_loss_func(pnts[:2, fr:to], A2, C2)
    # Выбор наименьшего значения функции потерь, возврат соответствующих ему параметров А и С
    if lf1 < lf2:
        return A1, C1, np.mean(((A1 * pnts[0, fr:to] - pnts[1, fr:to] + C1) ** 2) / (A1 ** 2 + 1))
    else:
        return A2, C2, np.mean(((A2 * pnts[0, fr:to] - pnts[1, fr:to] + C2) ** 2) / (A2 ** 2 + 1))


def firstPnt(pnts : np.ndarray) -> None:
    pnts[0, 0] = 0.5 * random.rand() - 0.25
    pnts[1, 0] = 0.5 * random.rand() - 0.25
    pnts[2, 0] = 2.0 * m.pi * random.rand() - m.pi


def createPnts(pnts : np.ndarray, N, d0 = 0.1, shape = 0, mess = 0.1) -> None:
    global pntsBuf

    i_ang = 0
    deltaAng = 0.2 * random.rand() - 0.

    for i in range(1, N):
        d = d0 * (1 + random.randn() * mess)
        pnts[0, i] = pnts[0, i - 1] + d * m.cos(pnts[2, i - 1])
        pnts[1, i] = pnts[1, i - 1] + d * m.sin(pnts[2, i - 1])

        if (shape == 0):    #polyline
            if (random.rand() > 1 - 5.0 / N): # 5 fractures in average
                pnts[2, i] = pnts[2, i - 1] + m.pi * random.rand() - m.pi/2
                i_ang = i
            else:
                pnts[2, i] = pnts[2, i_ang] * (1 + random.randn() * mess)
        elif (shape == 1):  #circle
            pnts[2, i] = pnts[2, i - 1] + deltaAng

    pntsBuf = pnts[:2, :].copy()

# def create_pnts(pnts : np.ndarray, N, mess=0.1, lidar_angle=pi, lidar_ax=(0.0, 0.0)) -> None:
#     global act_ostacles
#     lidar_ax_x = lidar_ax[0]
#     lidar_ax_y = lidar_ax[1]
#     R = 1  # радиус окружности, на которой выбираются центры окрестностей
#     w = 0.5  # ширина окрестности (половина стороны квадрата)
#     d_max = 2
#     seg_num = randint(7, 12)  # число сегментов (стены и препятствия)
#     segs = np.zeros([4, seg_num], dtype=float)
#     # act_ostacles = np.zeros(segs.shape, dtype=float)
#     # start_pt = None
#     c_x = R * cos((pi + lidar_angle) / 2)
#     c_y = R * sin((pi + lidar_angle) / 2)
#
#     st_x = c_x + uniform(-w, w)
#     st_y = c_y + uniform(-w, w)
#     for i in range(seg_num):
#         # st_x = c_x + uniform(-w, w)
#         # st_y = c_y + uniform(-w, w)
#
#         c_x = R * cos((pi + lidar_angle) / 2 - ((i + 1) * lidar_angle / seg_num))
#         c_y = R * sin((pi + lidar_angle) / 2 - ((i + 1) * lidar_angle / seg_num))
#
#         end_x = c_x + uniform(-w, w)
#         end_y = c_y + uniform(-w, w)
#
#         # if not start_pt:
#         #     st_x = c_x + uniform(-w, w)
#         #     st_y = c_y + uniform(-w, w)
#         #     start_pt = st_x, st_y
#         # st_x = c_x + uniform(-w, w)
#         # st_y = c_y + uniform(-w, w)
#         #
#         # end_x = c_x + uniform(-w, w)
#         # end_y = c_y + uniform(-w, w)
#
#         k = (end_y - st_y) / (end_x - st_x)
#         b = st_y - k * st_x
#
#         segs[0, i] = k
#         segs[1, i] = b
#         if st_x < end_x:
#             segs[2, i] = st_x
#             segs[3, i] = end_x
#         else:
#             segs[2, i] = end_x
#             segs[3, i] = st_x
#
#         st_x = end_x
#         st_y = end_y
#
#         act_ostacles = segs.copy()
#
#     delta_angle = lidar_angle / (N - 1)  # период измерения расстояния при вращении
#     # beam_angle = 0
#     y_min = min(lidar_ax_y, d_max * sin((pi + lidar_angle) / 2))
#     for i in range(N):
#         dist_noise = normalvariate(0.0, mess / 30)
#         beam_angle = (pi + lidar_angle) / 2 - i * delta_angle # + angle_noise
#         beam_k = tan(beam_angle)
#         beam_b = lidar_ax_y - beam_k * lidar_ax_x
#         d = m.inf
#         for j in range(seg_num):
#             intsec_x, intsec_y = calc_intersection(segs[0, j], segs[1, j], beam_k, beam_b)
#             if segs[2, j] <= intsec_x <= segs[3, j] and intsec_y > y_min:
#                 d_new = m.sqrt((intsec_x - lidar_ax_x) ** 2 + (intsec_y - lidar_ax_y) ** 2)
#                 if d_new < d:
#                     d = d_new
#                     nrst_intsec_x = intsec_x
#                     nrst_intsec_y = intsec_y
#
#         if m.isinf(d):
#             pnts[0, i] = d_max * cos(beam_angle)
#             pnts[1, i] = d_max * sin(beam_angle)
#         else:
#             pnts[0, i] = nrst_intsec_x  # Добавить шум
#             pnts[1, i] = nrst_intsec_y  # Добавить шум
#
#         pnts[0, i] = pnts[0, i] - dist_noise * sin(beam_angle - pi/2)
#         pnts[1, i] = pnts[1, i] + dist_noise * cos(beam_angle - pi/2)
#
#     pntsBuf = pnts[:2, :].copy()


def calc_intersection(A1, C1, A2, C2):
    # А это решение подстановкой, и это хотя бы работает
    x = (C2 - C1) / (A1 - A2)
    y = A2 * (C2 - C1) / (A1 - A2) + C2
    return x, y

def calc_normal(pnt, A1):
    # x0, y0 = pt
    x0 = pnt[0]
    y0 = pnt[1]
    A2 = -1/A1
    C2 = x0/A1 + y0

    return A2, C2


def getLines(lines: np.ndarray, pnts: np.ndarray, Npnts, tolerance=0.1, mode='lsm') -> int:
    """#returns the number of the gotten lines in lines"""
    t0 = time.time()
    global Nlines
    global timing

    # line = np.zeros([4])  #######################################################################################################
    line = np.zeros([2, 2], dtype=float)  # хранит 2 столбца с координатами х, у начальной и конечной точек отрезка
    pcross = np.array([0.0, 0.0])

    i = 1
    Nlines = 0
    # A = 0
    # C = 0
    # A_prev = 0
    # C_prev = 0

    while i < Npnts:
        gap = tolerance
        i0 = i

        while True:  # проход по всем точкам
            '''
            ЗДЕСЬ РЕАЛИЗУЕМ АППРОКСИМАЦИЮ МЕТОДАМИ - LSM В ЧАСТНОСТИ ПО ПЕРВЫМ ДВУМ ТОЧКАМ ОТРЕЗКАМ
            '''

            # # ОБНОВИТЬ КРИТЕРИЙ РАЗБИЕНИЯ - ЗАМЕНИТЬ УСРЕДНЕНИЕ НА ДРУГИЕ АППРОКСИМАЦИИ, СРАВНИТЬ ###############################
            # line[0] = (pnts[1, i] - pnts[1, i - 1]) / (pnts[0, i] - pnts[0, i - 1])
            # line[1] = pnts[1, i] - line[0] * pnts[0, i]
            # Новый формат - столбцы х, у
            # line[0] = pnts[0, i-1:i+1]  # координаты х двух текущих соседних точек
            # line[1] = pnts[1, i-1:i+1]  # координаты у двух текущих соседних точек
            line[:, 0] = pnts[:2, i-1]  # столбец координат начальной точки (х1 у1)
            line[:, 1] = pnts[:2, i]  # столбец координат конечной точки (х2 у2)

            # Вычисление k,b или A,C для определения расстояния
            # A = (line[1, 1] - line[1, 0]) / (line[0, 1] - line[0, 0])
            # C = line[1, 0] - A * line[0, 0]
            A, C, q0 = line_approx_lsm(pnts, i-1, i+1)

            byNpnts = 2

            while True:  # проход по найденному отрезку - поиск конца  ================== РАЗОБРАТЬСЯ С ИНДЕКСАМИ =====================================
                
                i += 1
                '''
                ЗДЕСЬ ЭТОТ ЗАМЕЧАТЕЛЬНЫЙ КРИТЕРИЙ РЕДАКТИРУЕМ
                РАЗОБРАТЬСЯ ЧТО ОН ЗНАЧИТ И ДЛЯ ЧЕГО НУЖЕН: КАК Я ПОЛАГАЮ, ЗДЕСЬ У НАС ВЫСЧИТЫВАЕТСЯ ЧИСЛО ТОЧЕК
                ПРОИСХОДИТ ПРОВЕРКА НА УДАЛЁННОСТЬ СЛЕДУЮЩЕЙ ТОЧКИ
                '''

                # if (i < N and abs(line[0] * pnts[0, i] - pnts[1, i] + line[1]) / m.sqrt(line[0]**2 + 1) < gap): ####################################
                #     if (not byNpnts % 2):
                #         line[0] = (pnts[1, i - byNpnts // 2] - pnts[1, i - byNpnts]) / (pnts[0, i - byNpnts // 2] - pnts[0, i - byNpnts])
                #         line[1] = pnts[1, i - byNpnts] - line[0] * pnts[0, i - byNpnts]
                #     byNpnts += 1
                if i < N and abs(A * pnts[0, i] - pnts[1, i] + C) / m.sqrt(A ** 2 + 1) < gap:  # если есть следующая точка и tolerance не превышен
                    A, C, q0 = line_approx_lsm(pnts, i-byNpnts, i+1)  # включаем в аппроксимацию
                    byNpnts += 1

                else:
                    # # аппроксимация в зависимости от режима mode (avg - усреднение, lsm - МНК)  #######################################
                    # if mode == 'avg':
                    #     (line[0], line[1], q0) = lineApproxAveraging(pnts, i - byNpnts, i)
                    # elif mode == 'lsm':
                    #     (line[0], line[1], q0) = line_approx_lsm(pnts, i - byNpnts, i)
                    # elif mode == 'opt_lsm':
                    #     (line[0], line[1], q0) = line_approx_opt_lsm(pnts, i - byNpnts, i)
                    while (q0 > 0.0001):  # поиск оптимальной концевой точки, чтобы забрать лишних следующих ###############################
                        # if mode == 'avg':  ####################################################################################3
                        #     (line_0, line_1, q) = lineApproxAveraging(pnts, i - byNpnts, i - 1)
                        # elif mode == 'lsm':
                        #     (line_0, line_1, q) = line_approx_lsm(pnts, i - byNpnts, i - 1)
                        # elif mode == 'opt_lsm':
                        #     (line_0, line_1, q) = line_approx_lsm(pnts, i - byNpnts, i - 1)
                        A_opt, C_opt, q = line_approx_lsm(pnts, i-byNpnts, i-1)
                        if (q > q0):  # если увеличилось среднее отклонение, заканчиваем (продолжаем с последним оптимумом)
                            break
                        else:  # сохраняем текущий оптимум
                            i -= 1
                            byNpnts -= 1
                            # line[0] = line_0  ####################################################################################
                            # line[1] = line_1
                            A = A_opt
                            C = C_opt

                            q0 = q
                            # if byNpnts < 2:
                            #     break

                    # работаем с полученными А и С
                    '''
                    ВОТ ЗДЕСЬ ТОЖЕ РАЗОБРАТЬСЯ ЧТО ЭТО ПРОИСХОДИТ
                    НУ ЗДЕСЬ ТОЧКИ ПЕРЕСЕЧЕНИЯ ВРОДЕ
                    '''

                    if Nlines > 0:  # если уже найден хотя бы один луч - ищем пересечение текущей прямой с ним
                        # pcross[0] = (line[1] - lines[1, Nlines - 1]) / (lines[0, Nlines - 1] - line[0]) ###############################
                        # pcross[1] = line[0] * pcross[0] + line[1]
                        pcross[0] = (C - C_prev) / (A_prev - A)
                        pcross[1] = A * pcross[0] + C

                        # line[0, 0], line[1, 0] = calc_intersection(A_prev, C_prev, A, C) # Это моё

                        # if np.linalg.norm(pnts[:2, i - byNpnts] - pcross) > tolerance or m.isnan(pcross[0]) or m.isinf(pcross[0]):  #####################
                        #     if (byNpnts <= 2):
                        #         pcross[0] = (pnts[0, i - 2] + lines[0, Nlines - 1] * pnts[1, i - 2] - lines[0, Nlines - 1] * lines[1, Nlines - 1]) / (lines[0, Nlines - 1]**2 + 1)
                        #         pcross[1] = lines[0, Nlines - 1] * pcross[0] + lines[1, Nlines - 1]
                        #
                        #         line[0] = (pnts[1, i - 1] - pcross[1]) / (pnts[0, i - 1] - pcross[0])
                        #         line[1] = pcross[1] - line[0] * pcross[0]
                        #         lines[3, Nlines - 1] = pcross[0]
                        #         line[2] = pcross[0]
                        #     else:
                        #         i = i0
                        #         gap *= 0.75
                        #         break
                        # else:
                        #     lines[3, Nlines - 1] = pcross[0]
                        #     line[2] = pcross[0]

                        if np.linalg.norm(pnts[:2, i - byNpnts] - pcross) > tolerance or m.isnan(pcross[0]) or m.isinf(pcross[0]):
                            if (byNpnts <= 2):
                                pcross[0] = (pnts[0, i - 2] + A_prev * pnts[1, i - 2] - A_prev * C_prev) / (A_prev**2 + 1)
                                pcross[1] = A_prev * pcross[0] + C_prev

                                # line[0] = (pnts[1, i - 1] - pcross[1]) / (pnts[0, i - 1] - pcross[0])
                                # line[1] = pcross[1] - line[0] * pcross[0]
                                # lines[3, Nlines - 1] = pcross[0]
                                # line[2] = pcross[0]
                                line[0, 0] = pcross[0]
                                line[1, 0] = pcross[1]

                            else:
                                i = i0
                                gap *= 0.75
                                break
                        else:
                            # lines[3, Nlines - 1] = pcross[0]
                            # line[2] = pcross[0]
                            line[0, 0] = pcross[0]
                            line[1, 0] = pcross[1]

                    else:  # если ещё не нашли линий - пересечение прямой с нормалью из первой точки датасета (ГОТОВО)
                        # line[2] = (pnts[0, 0] + line[0] * pnts[1, 0] - line[0] * line[1]) / (line[0]**2 + 1) #################
                        A_prev, C_prev = calc_normal(pnts[:2, 0], A)
                        line[0, 0], line[1, 0] = calc_intersection(A_prev, C_prev, A, C)
                        # =========== ПОЛУЧАЕМ ЛУЧ, ИСХОДЯЩИЙ ИЗ ЭТОЙ ТОЧКИ (КОНЕЦ ПОКА НЕ ИЗВЕСТЕН) ==============

                    if i > N - 1:  # если точка последняя - пересечение прямой с нормалью из последней точки датасета (ГОТОВО)
                        # line[3] = (pnts[0, N - 1] + line[0] * pnts[1, N - 1] - line[0] * line[1]) / (line[0]**2 + 1)  #########
                        A_last, C_last = calc_normal(pnts[:2, N-1], A)
                        line[0, 1], line[1, 1] = calc_intersection(A, C, A_last, C_last)

                    break

            A_prev = A
            C_prev = C

            if (i > i0):
                break
            else:
                continue

        #### ВОТ ТУТ С ПРИСВОЕНИЕМ ПОРАБОТАТЬ - МОЖЕМ НАЧАЛО ОТРЕЗКА ВНЕСТИ, КОНЕЦ ПОТОМ ВНЕСЁМ В КАЧЕСТВЕ НАЧАЛА СЛЕДУЮЩЕГО
        # ТАКЖЕ МОЖЕМ ВНЕСТИ САМУЮ ПОСЛЕДНЮЮ ТОЧКУ
        lines[:, Nlines] = line[:, 0]
        Nlines += 1
        if i > N - 1: # если последняя точка рассмотрена, включаем второй столбец line (поскольку для последнего отрезка определён конец)
            lines[:, Nlines] = line[:, 1]

    t1 = time.time()

    # сохранение значений времени выполнения аппроксимации всей траектории
    if mode == 'avg':
        timing[0] = t1 - t0
    if mode == 'lsm':
        timing[1] = t1 - t0
    elif mode == 'opt_lsm':
        timing[2] = t1 - t0

    return Nlines


def drawLoad(xlim=(-4, 4), ylim=(-4, 4)):
    ax.cla()

    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect('equal')
    # ax.scatter(0, 0, color='black')
    ax.scatter(pnts[0, 0], pnts[1, 0], s=10, marker='o', color='red')
    ax.scatter(pnts[0, 1:], pnts[1, 1:], s=10, marker='o', color='gray')
    ax.set(title=mode)

    # plot_obstacles(ax)
    #
    # for i in range(N):
    #     ax.plot([lidar_ax[0], pnts[0, i]], [lidar_ax[1], pnts[1, i]], color='cyan', linewidth=0.2)

    # for i in range(Nlines):
    #     ax.plot([lines[2, i], lines[3, i]], [lines[0, i] * lines[2, i] + lines[1, i], lines[0, i] * lines[3, i] + lines[1, i]], linewidth=1.5) ###########
    for i in range(Nlines):
        ax.plot([lines[0, i], lines[0, i+1]], [lines[1, i], lines[1, i+1]],linewidth=2)

    ax.axis('scaled')

    fig.canvas.draw_idle()

def plot_obstacles(ax):
    for i in range(act_ostacles.shape[1]):
        ax.plot([act_ostacles[2, i], act_ostacles[3, i]],
                [act_ostacles[0, i] * act_ostacles[2, i] + act_ostacles[1, i],
                 act_ostacles[0, i] * act_ostacles[3, i] + act_ostacles[1, i]],
                linewidth=1.5)


def nextPnts(event):
    with mutex:
        firstPnt(pnts)

        createPnts(pnts, N, shape=shape, mess=mess)
        # create_pnts(pnts, N)
        getLines(lines, pnts, N, tol, mode=mode)

        drawLoad()


def updatePnts(val):
    global mess
    with mutex:
        mess = val
        createPnts(pnts, N, shape=shape, mess=mess)
        # create_pnts(pnts, N)
        getLines(lines, pnts, N, tol, mode=mode)
        drawLoad()


def updateLinesTolerance(val):
    global tol
    
    with mutex:
        tol = val
        getLines(lines, pnts, N, tol, mode=mode)
    
    drawLoad(ax.get_xlim(), ax.get_ylim())


def updatePntsShape(event):
    global shape
    with mutex:
        shape += 1
        if shape > 1:
            shape = 0
        createPnts(pnts, N, shape=shape, mess=mess)
        # create_pnts(pnts, N, mess=mess)
        getLines(lines, pnts, N, tol, mode=mode)
        drawLoad()


jit = False
def jitter(event):
    global jit

    def foo():
        while jit and plt.get_fignums():
            with mutex:
                rns = np.zeros([2, N])
                for i in range(N):
                    if random.rand() > 0.9:
                        rns[:, i] += 0.5 * random.rand(2) - 0.25
                pnts[:2, :] = pntsBuf + 0.02 * random.rand(2, N) - 0.01 + rns

                getLines(lines, pnts, N, tol, mode=mode)
                drawLoad(ax.get_xlim(), ax.get_ylim())
            time.sleep(0.5)

    with mutex:
        jit = not jit
        threading.Thread(target=foo).start()


# Смена метода аппроксимации: avg - усреднение, lsm - МНК
def change_mode(event):
    global mode
    with mutex:
        if mode == 'avg':
            mode = 'lsm'
        elif mode == 'lsm':
            mode = 'avg'
        # elif mode == 'opt_lsm':
        #     mode = 'avg'
        getLines(lines, pnts, N, tol, mode=mode)
        drawLoad()


# Вывод графиков времени аппроксимации каждым алгоритмом
def plot_timing_cumsum(event):
    tm_fig, tm_ax = plt.subplots()
    t_avg_cumsum = 0  # кумулятивная сумма для усреднения
    t_lsm_cumsum = 0  # кумулятивная сумма для МНК
    n_samples = 50  # число различных наборов точек
    n_tests = 10  # число запусков на одном наборе
    with mutex:
        for i in range(n_samples):
            nextPnts(None)
            for j in range(n_tests):
                # дважды меняем метод аппроксимации
                change_mode(None)
                change_mode(None)
                tm_ax.plot([j + n_tests * i, j + n_tests * i + 1],
                           [t_avg_cumsum, t_avg_cumsum + timing[0]],
                           color='red')
                tm_ax.plot([j + n_tests * i, j + n_tests * i + 1],
                           [t_lsm_cumsum, t_lsm_cumsum + timing[1]],
                           color='blue')
                t_avg_cumsum += timing[0]
                t_lsm_cumsum += timing[1]

        tm_fig.show()


def main():

    firstPnt(pnts)
    createPnts(pnts, N, shape=shape, mess=mess)
    # create_pnts(pnts, N, mess=mess)

    getLines(lines, pnts, N, tol, mode=mode)
    drawLoad()

    ax1 = plt.axes([0.15, 0.17, 0.45, 0.03])
    ax2 = plt.axes([0.15, 0.14, 0.45, 0.03])
    ax3 = plt.axes([0.55, 0.28, 0.1, 0.04])
    ax4 = plt.axes([0.55, 0.35, 0.1, 0.04])
    ax5 = plt.axes([0.55, 0.42, 0.1, 0.04])
    ax6 = plt.axes([0.55, 0.49, 0.1, 0.04])
    ax7 = plt.axes([0.55, 0.6, 0.1, 0.1])

    sz1 = Slider(ax1, 'tolerance', 0.0, 0.8, tol, valstep=0.02)
    sz1.on_changed(updateLinesTolerance)

    sz2 = Slider(ax2, 'mess', 0.0, 1.0, mess, valstep=0.02)
    sz2.on_changed(updatePnts)

    btn1 = Button(ax3, 'Jitter', hovercolor='0.975')
    btn1.on_clicked(jitter)

    btn2 = Button(ax4, 'Shape', hovercolor='0.975')
    btn2.on_clicked(updatePntsShape)

    btn3 = Button(ax5, 'Next', hovercolor='0.975')
    btn3.on_clicked(nextPnts)

    btn4 = Button(ax6, 'Mode', hovercolor='0.975')
    btn4.on_clicked(change_mode)

    btn5 = Button(ax7, 'Start\ntest', hovercolor='0.975')
    btn5.on_clicked(plot_timing_cumsum)

    plt.show()


if __name__ == "__main__":
    main()
