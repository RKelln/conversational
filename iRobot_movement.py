import asyncio
import random
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import Create3, event

# Initialize the Bluetooth connection
bluetooth = Bluetooth()
robot = Create3(bluetooth)

is_docking = False
speed = 0
obstacle_threshold = 40

async def forward(robot):
    await robot.set_lights_on_rgb(0, 255, 0)
    await robot.set_wheel_speeds(speed, speed)

async def backoff(robot):
    await robot.set_lights_on_rgb(255, 80, 0)
    await robot.set_wheel_speeds(-speed, -speed)
    await asyncio.sleep(1)
    turn_direction = random.choice(['left', 'right'])
    if turn_direction == 'left':
        await robot.turn_left(120)
    else:
        await robot.turn_right(120)

def front_obstacle(sensors):
    return sensors[3] > obstacle_threshold

async def check_battery_and_dock(robot):
    global is_docking
    while True:
        battery = await robot.get_battery_level()
        battery_percentage = battery[1]
        print(f"Battery level: {battery_percentage}%")

        if battery_percentage < 5 and not is_docking:
            print("Battery low! Returning to dock.")
            is_docking = True
            await robot.dock()
            print("Docking...")

            while True:
                battery = await robot.get_battery_level()
                battery_percentage = battery[1]
                print(f"Charging... Battery level: {battery_percentage}%")
                if battery_percentage >= 100:
                    break
                await asyncio.sleep(30)

            print("Battery fully charged! Resuming activities.")
            is_docking = False
            await robot.undock()
            print("Undocking...")

        await asyncio.sleep(30)

@event(robot.when_play)
async def play(robot):
    print("Starting play event...")
    try:
        asyncio.create_task(check_battery_and_dock(robot))

        await forward(robot)
        while True:
            if is_docking:
                await asyncio.sleep(0.1)
                continue

            try:
                sensors = await robot.get_ir_proximity()
                sensors_values = sensors.sensors
                print(f"Sensors: {sensors_values}")
                if front_obstacle(sensors_values):
                    await backoff(robot)
                    await forward(robot)
            except Exception as e:
                print(f"Error reading sensors: {e}")
                await asyncio.sleep(1)

            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")

@event(robot.when_bumped, [True, True])
async def handle_bump(robot):
    if is_docking:
        return

    print("Collision detected by bumper! Reversing and turning.")
    await backoff(robot)
    await forward(robot)

robot.play()
