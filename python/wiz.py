import config
import subprocess
import numpy as np

def change_bulb_color(rgb: np.array):
  """
  Changes the color of the WiZ bulbs in the background.

  Args:
      rgb: A NumPy array representing the desired color (red, green, blue).
  """
  r = (min(max(0, int(rgb[0])), 255))  # Clamp color values
  g = (min(max(0, int(rgb[1])), 255))
  b = (min(max(0, int(rgb[2])), 255))

  processes = []
  for IP in config.WIZ_IPS:
    command = ["riz", "-c", "%s,%s,%s" % (r, g, b), IP]
    # Run the command in the background using Popen
    process = subprocess.Popen(command)
    processes.append(process)

  # Wait for all background processes to finish (optional)
  # for process in processes:
  #   process.wait()

  return processes  # Optional: return the list of processes for monitoring
    

def change_bulb_speed(speed: int):

    WIZ_MIN_SPEED = 20
    WIZ_MAX_SPEED = 200

    speed = 20

    for IP in config.WIZ_IPS:
        subprocess.run(["riz", "-p",str(min(max(WIZ_MIN_SPEED, int(speed)), WIZ_MAX_SPEED)), IP])