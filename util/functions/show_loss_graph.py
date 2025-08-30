import matplotlib.pyplot as plt
def show_loss_graph(losses_list):

    # サンプルデータ
    # losses_list = [1.50, 1.25, 1.05, 0.98, 0.92, 0.85, 0.78, 0.70, 0.65, 0.60]

    # グラフを描画
    plt.plot(losses_list, marker='None')

    # タイトルとラベルを追加
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')

    # グリッドを追加
    plt.grid(True)

    # グラフを表示
    plt.show()