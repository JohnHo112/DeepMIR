import matplotlib.pyplot as plt

def plot_loss_acc(train_loss, train_acc, val_loss, val_acc):
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.title("loss")
    plt.savefig("task2/results/loss.png")
    plt.close()

    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="val")
    plt.legend()
    plt.title("acc")
    plt.ylim((0, 1))
    plt.savefig("task2/results/acc.png")
    plt.close()

artist2idx = {
    'aerosmith': 0, 
    'beatles': 1, 
    'creedence_clearwater_revival': 2, 
    'cure': 3, 
    'dave_matthews_band': 4, 
    'depeche_mode': 5, 
    'fleetwood_mac': 6, 
    'garth_brooks': 7, 
    'green_day': 8, 
    'led_zeppelin': 9, 
    'madonna': 10, 
    'metallica': 11, 
    'prince': 12, 
    'queen': 13, 
    'radiohead': 14, 
    'roxette': 15, 
    'steely_dan': 16, 
    'suzanne_vega': 17, 
    'tori_amos': 18, 
    'u2': 19
    }
idx2artist = {
    0: 'aerosmith', 
    1: 'beatles', 
    2: 'creedence_clearwater_revival', 
    3: 'cure', 
    4: 'dave_matthews_band', 
    5: 'depeche_mode', 
    6: 'fleetwood_mac', 
    7: 'garth_brooks', 
    8: 'green_day', 
    9: 'led_zeppelin', 
    10: 'madonna', 
    11: 'metallica', 
    12: 'prince', 
    13: 'queen', 
    14: 'radiohead', 
    15: 'roxette', 
    16: 'steely_dan', 
    17: 'suzanne_vega', 
    18: 'tori_amos', 
    19: 'u2' 
}