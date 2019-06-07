from matplotlib import pyplot as plt


def mk_fig1(data):
    plt.figure()
    plt.plot(data.bidopen)
    plt.plot(data.bidclose)
    plt.plot(data.bidlow)
    plt.plot(data.bidhigh)


def mk_fig2(data):
    plt.figure()
    plt.plot(data.bidhigh - data.bidlow)
    plt.plot(data.bidclose - data.bidlow)


def mk_fig3(data):
    plt.figure()
    plt.plot(data.bidhigh - data.bidlow)
    plt.plot(data.bidhigh - data.bidopen)


def mk_fig4(data):
    plt.figure()
    plt.plot(data.bidclose - data.bidopen)


def mk_fig5(data):
    plt.figure()
    plt.plot(data.tickqty)


def mk_all(data):
    mk_fig1(data)
    mk_fig2(data)
    mk_fig3(data)
    mk_fig4(data)
    mk_fig5(data)
