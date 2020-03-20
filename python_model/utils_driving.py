import os
import math
import time
import pyglet
import pickle
import numpy as np
import theano as th
import matplotlib.cm
import pyglet.gl as gl
import theano.tensor as tt
import pyglet.graphics as graphics
from pyglet.window import key



class Feature(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, *args):
        return self.f(*args)
    def __add__(self, r):
        return Feature(lambda *args: self(*args)+r(*args))
    def __radd__(self, r):
        return Feature(lambda *args: r(*args)+self(*args))
    def __mul__(self, r):
        return Feature(lambda *args: self(*args)*r)
    def __rmul__(self, r):
        return Feature(lambda *args: r*self(*args))
    def __pos__(self, r):
        return self
    def __neg__(self):
        return Feature(lambda *args: -self(*args))
    def __sub__(self, r):
        return Feature(lambda *args: self(*args)-r(*args))
    def __rsub__(self, r):
        return Feature(lambda *args: r(*args)-self(*args))

def feature(f):
    return Feature(f)

def speed(s=1.):
    @feature
    def f(t, x, u):
        return -(x[3]-s)*(x[3]-s)
    return f

def control():
    @feature
    def f(t, x, u):
        return -u[0]**2-u[1]**2
    return f

# def bounded_control(bounds, width=0.05):
#     @feature
#     def f(t, x, u):
#         ret = 0.
#         for i, (a, b) in enumerate(bounds):
#             return -tt.exp((u[i]-b)/width)-tt.exp((a-u[i])/width)
#     return f

class World(object):
    def __init__(self):
        self.cars = []
        self.lanes = []
        self.roads = []
        self.fences = []
        self.objects = []
    def simple_reward(self, trajs=None, lanes=None, roads=None, fences=None, speed=1., speed_import=1.):
        if lanes is None:
            lanes = self.lanes
        if roads is None:
            roads = self.roads
        if fences is None:
            fences = self.fences
        if trajs is None:
            trajs = [c.linear for c in self.cars]
        elif isinstance(trajs, Car):
            trajs = [c.linear for c in self.cars if c!=trajs]
        r = 0.1*feature.control()
        theta = [1., -50., 10., 10., -60.] # Simple model
        # theta = [.959, -46.271, 9.015, 8.531, -57.604]
        for lane in lanes:
            r = r+theta[0]*lane.gaussian()
        for fence in fences:
            r = r+theta[1]*fence.gaussian()
        for road in roads:
            r = r+theta[2]*road.gaussian(10.)
        if speed is not None:
            r = r+speed_import*theta[3]*feature.speed(speed)
        for traj in trajs:
            r = r+theta[4]*traj.gaussian()
        return r

class Trajectory(object):
    def __init__(self, T, dyn):
        self.dyn = dyn
        self.T = T
        self.x0 = th.shared(np.zeros(dyn.nx))
        self.u = [th.shared(np.zeros(dyn.nu)) for _ in range(self.T)]
        self.x = []
        z = self.x0
        for t in range(T):
            z = dyn(z, self.u[t])
            self.x.append(z)
        self.next_x = th.function([], self.x[0])
    def tick(self):
        self.x0.set_value(self.next_x())
        for t in range(self.T-1):
            self.u[t].set_value(self.u[t+1].get_value())
        self.u[self.T-1].set_value(np.zeros(self.dyn.nu))

class Dynamics(object):
    def __init__(self, nx, nu, f, dt=None):
        self.nx = nx
        self.nu = nu
        self.dt = dt
        if dt is None:
            self.f = f
        else:
            self.f = lambda x, u: x+dt*f(x, u)
    def __call__(self, x, u):
        return self.f(x, u)

class CarDynamics(Dynamics):
    def __init__(self, dt=0.1, ub=[(-3., 3.), (-1., 1.)], friction=1.):
        def f(x, u):
            return tt.stacklists([
                x[3]*tt.cos(x[2]),
                x[3]*tt.sin(x[2]),
                x[3]*u[0],
                u[1]-x[3]*friction
            ])
        Dynamics.__init__(self, 4, 2, f, dt)

class Lane(object): pass

class StraightLane(Lane):
    def __init__(self, p, q, w):
        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w
        self.m = (self.q-self.p)/np.linalg.norm(self.q-self.p)
        self.n = np.asarray([-self.m[1], self.m[0]])
    def shifted(self, m):
        return StraightLane(self.p+self.n*self.w*m, self.q+self.n*self.w*m, self.w)
    def dist2(self, x):
        r = (x[0]-self.p[0])*self.n[0]+(x[1]-self.p[1])*self.n[1]
        return r*r
    def gaussian(self, width=0.5):
        @feature.feature
        def f(t, x, u):
            return tt.exp(-0.5*self.dist2(x)/(width**2*self.w*self.w/4.))
        return f

class Simulation(object):
    def __init__(self, name, total_time=1000, recording_time=[0,1000]):
        self.name = name.lower()
        self.total_time = total_time
        self.recording_time = [max(0,recording_time[0]), min(total_time,recording_time[1])]
        self.frame_delay_ms = 0

    def reset(self):
        self.trajectory = []
        self.alreadyRun = False
        self.ctrl_array = [[0]*self.input_size]*self.total_time

    @property
    def ctrl(self):
        return self.ctrl_array
    @ctrl.setter
    def ctrl(self, value):
        self.reset()
        self.ctrl_array = value.copy()
        self.run(reset=False)

class DrivingSimulation(Simulation):
    def __init__(self, name, total_time=50, recording_time=[0,50]):
        super(DrivingSimulation, self).__init__(name, total_time=total_time, recording_time=recording_time)
        self.world = World()
        clane = StraightLane([0., -1.], [0., 1.], 0.17)
        self.world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
        self.world.roads += [clane]
        self.world.fences += [clane.shifted(2), clane.shifted(-2)]
        self.dyn = CarDynamics(0.1)
        self.robot = Car(self.dyn, [0., -0.3, np.pi/2., 0.4], color='orange')
        self.human = Car(self.dyn, [0.17, 0., np.pi/2., 0.41], color='white')
        self.world.cars.append(self.robot)
        self.world.cars.append(self.human)
        self.initial_state = [self.robot.x, self.human.x]
        self.input_size = 2
        self.reset()
        self.viewer = None

    def initialize_positions(self):
        self.robot_history_x = []
        self.robot_history_u = []
        self.human_history_x = []
        self.human_history_u = []
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]

    def reset(self):
        super(DrivingSimulation, self).reset()
        self.initialize_positions()

    def run(self, reset=False):
        if reset:
            self.reset()
        else:
            self.initialize_positions()
        for i in range(self.total_time):
            self.robot.u = self.ctrl_array[i]
            if i < self.total_time//5:
                self.human.u = [0, self.initial_state[1][3]]
            elif i < 2*self.total_time//5:
                self.human.u = [1., self.initial_state[1][3]]
            elif i < 3*self.total_time//5:
                self.human.u = [-1., self.initial_state[1][3]]
            elif i < 4*self.total_time//5:
                self.human.u = [0, self.initial_state[1][3]*1.3]
            else:
                self.human.u = [0, self.initial_state[1][3]*1.3]
            self.robot_history_x.append(self.robot.x)
            self.robot_history_u.append(self.robot.u)
            self.human_history_x.append(self.human.x)
            self.human_history_u.append(self.human.u)
            self.robot.move()
            self.human.move()
            self.trajectory.append([self.robot.x, self.human.x])
        self.alreadyRun = True

    # I keep all_info variable for the compatibility with mujoco wrapper
    def get_trajectory(self, all_info=True):
        if not self.alreadyRun:
            self.run()
        return self.trajectory.copy()

    def get_recording(self, all_info=True):
        traj = self.get_trajectory(all_info=all_info)
        return traj[self.recording_time[0]:self.recording_time[1]]

    def watch(self, repeat_count=1, name="unnamed", loc=None):
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]
        if self.viewer is None:
            self.viewer = Visualizer(0.1, magnify=1.2, name=name, loc=loc)
            self.viewer.main_car = self.robot
            self.viewer.use_world(self.world)
            self.viewer.paused = False
        for _ in range(repeat_count):
            self.viewer.run_modified(history_x=[self.robot_history_x, self.human_history_x], history_u=[self.robot_history_u, self.human_history_u])
        self.viewer.window.close()
        self.viewer = None


class Driver(DrivingSimulation):
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driver', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30*np.min([np.square(recording[:,0,0]-0.17), np.square(recording[:,0,0]), np.square(recording[:,0,0]+0.17)], axis=0)))

        # keeping speed (lower is better)
        keeping_speed = np.mean(np.square(recording[:,0,3]-1))

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:,0,2]))

        # collision avoidance (lower is better)
        collision_avoidance = np.mean(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1]))))

        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

    @property
    def state(self):
        return [self.robot.x, self.human.x]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)

class Car(object):
    def __init__(self, dyn, x0, color='yellow', T=5):
        self.data0 = {'x0': x0}
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.T = T
        self.dyn = dyn
        self.traj = Trajectory(T, dyn)
        self.traj.x0.set_value(x0)
        self.linear = Trajectory(T, dyn)
        self.linear.x0.set_value(x0)
        self.color = color
        self.default_u = np.zeros(self.dyn.nu)
    def reset(self):
        self.traj.x0.set_value(self.data0['x0'])
        self.linear.x0.set_value(self.data0['x0'])
        for t in range(self.T):
            self.traj.u[t].set_value(np.zeros(self.dyn.nu))
            self.linear.u[t].set_value(self.default_u)
    def move(self):
        self.traj.tick()
        self.linear.x0.set_value(self.traj.x0.get_value())
    @property
    def x(self):
        return self.traj.x0.get_value()
    @x.setter
    def x(self, value):
        self.traj.x0.set_value(value)
    @property
    def u(self):
        return self.traj.u[0].get_value()
    @u.setter
    def u(self, value):
        self.traj.u[0].set_value(value)
    def control(self, steer, gas):
        pass


class Visualizer(object):
    def __init__(self, dt=0.5, fullscreen=False, name='unnamed', iters=1000, magnify=1., loc=None):
        self.autoquit = True
        self.frame = None
        self.subframes = None
        self.visible_cars = []
        self.magnify = magnify
        self.camera_center = None
        self.name = name
        self.output = None
        self.iters = iters
        self.objects = []
        self.event_loop = pyglet.app.EventLoop()
        self.window = pyglet.window.Window(600, 600, fullscreen=fullscreen, caption=name)
        if loc is not None:
            self.window.set_location(*loc)
        self.grass = pyglet.resource.texture('imgs/grass.png')
        self.window.on_draw = self.on_draw
        self.lanes = []
        self.cars = []
        self.dt = dt
        self.anim_x = {}
        self.prev_x = {}
        self.feed_u = None
        self.feed_x = None
        self.prev_t = None
        self.joystick = None
        self.keys = key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.on_key_press = self.on_key_press
        self.main_car = None
        self.heat = None
        self.heatmap = None
        self.heatmap_valid = False
        self.heatmap_show = False
        self.cm = matplotlib.cm.jet
        self.paused = False
        self.label = pyglet.text.Label(
            'Speed: ',
            font_name='Times New Roman',
            font_size=24,
            x=30, y=self.window.height - 30,
            anchor_x='left', anchor_y='top'
        )

        def centered_image(filename):
            img = pyglet.resource.image(filename)
            img.anchor_x = img.width / 2.
            img.anchor_y = img.height / 2.
            return img

        def car_sprite(color, scale=0.15 / 600.):
            sprite = pyglet.sprite.Sprite(centered_image('imgs/car-{}.png'.format(color)), subpixel=True)
            sprite.scale = scale
            return sprite

        def object_sprite(name, scale=0.15 / 600.):
            sprite = pyglet.sprite.Sprite(centered_image('imgs/{}.png'.format(name)), subpixel=True)
            sprite.scale = scale
            return sprite

        self.sprites = {c: car_sprite(c) for c in ['red', 'yellow', 'purple', 'white', 'orange', 'gray', 'blue']}
        self.obj_sprites = {c: object_sprite(c) for c in ['cone', 'firetruck']}

    def use_world(self, world):
        self.cars = [c for c in world.cars]
        self.lanes = [c for c in world.lanes]
        self.objects = [c for c in world.objects]

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.event_loop.exit()
        if symbol == key.SPACE:
            self.paused = not self.paused
        if symbol == key.D:
            self.reset()

    def control_loop(self, _=None):
        # print "Time: ", time.time()
        if self.paused:
            return
        if self.iters is not None and len(self.history_x[0]) >= self.iters:
            if self.autoquit:
                self.event_loop.exit()
            return
        if self.feed_u is not None and len(self.history_u[0]) >= len(self.feed_u[0]):
            if self.autoquit:
                self.event_loop.exit()
            return
        if self.pause_every is not None and self.pause_every > 0 and len(self.history_u[0]) % self.pause_every == 0:
            self.paused = True
        steer = 0.
        gas = 0.
        if self.keys[key.UP]:
            gas += 1.
        if self.keys[key.DOWN]:
            gas -= 1.
        if self.keys[key.LEFT]:
            steer += 1.5
        if self.keys[key.RIGHT]:
            steer -= 1.5
        if self.joystick:
            steer -= self.joystick.x * 3.
            gas -= self.joystick.y
        self.heatmap_valid = False
        for car in self.cars:
            self.prev_x[car] = car.x
        if self.feed_u is None:
            for car in reversed(self.cars):
                car.control(steer, gas)
        else:
            for car, fu, hu in zip(self.cars, self.feed_u, self.history_u):
                car.u = fu[len(hu)]
        for car, hist in zip(self.cars, self.history_u):
            hist.append(car.u)
        for car in self.cars:
            car.move()
        for car, hist in zip(self.cars, self.history_x):
            hist.append(car.x)
        self.prev_t = time.time()

    def center(self):
        if self.main_car is None:
            return np.asarray([0., 0.])
        elif self.camera_center is not None:
            return np.asarray(self.camera_center[0:2])
        else:
            return self.anim_x[self.main_car][0:2]

    def camera(self):
        o = self.center()
        gl.glOrtho(o[0] - 1. / self.magnify, o[0] + 1. / self.magnify, o[1] - 1. / self.magnify,
                   o[1] + 1. / self.magnify, -1., 1.)

    def set_heat(self, f):
        x = th.shared(np.zeros(4))
        u = th.shared(np.zeros(2))
        func = th.function([], f(0, x, u))

        def val(p):
            x.set_value(np.asarray([p[0], p[1], 0., 0.]))
            return func()

        self.heat = val

    def draw_heatmap(self):
        if not self.heatmap_show:
            return
        SIZE = (256, 256)
        if not self.heatmap_valid:
            o = self.center()
            x0 = o - np.asarray([1.5, 1.5]) / self.magnify
            x0 = np.asarray([x0[0] - x0[0] % (1. / self.magnify), x0[1] - x0[1] % (1. / self.magnify)])
            x1 = x0 + np.asarray([4., 4.]) / self.magnify
            x0 = o - np.asarray([1., 1.]) / self.magnify
            x1 = o + np.asarray([1., 1.]) / self.magnify
            self.heatmap_x0 = x0
            self.heatmap_x1 = x1
            vals = np.zeros(SIZE)
            for i, x in enumerate(np.linspace(x0[0], x1[0], SIZE[0])):
                for j, y in enumerate(np.linspace(x0[1], x1[1], SIZE[1])):
                    vals[j, i] = self.heat(np.asarray([x, y]))
            vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals) + 1e-6)
            vals = self.cm(vals)
            vals[:, :, 3] = 0.7
            vals = (vals * 255.99).astype('uint8').flatten()
            vals = (gl.GLubyte * vals.size)(*vals)
            img = pyglet.image.ImageData(SIZE[0], SIZE[1], 'RGBA', vals, pitch=SIZE[1] * 4)
            self.heatmap = img.get_texture()
            self.heatmap_valid = True
        gl.glClearColor(1., 1., 1., 1.)
        gl.glEnable(self.heatmap.target)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindTexture(self.heatmap.target, self.heatmap.id)
        gl.glEnable(gl.GL_BLEND)
        x0 = self.heatmap_x0
        x1 = self.heatmap_x1
        graphics.draw(4, gl.GL_QUADS,
                      ('v2f', (x0[0], x0[1], x1[0], x0[1], x1[0], x1[1], x0[0], x1[1])),
                      ('t2f', (0., 0., 1., 0., 1., 1., 0., 1.)),
                      # ('t2f', (0., 0., SIZE[0], 0., SIZE[0], SIZE[1], 0., SIZE[1]))
                      )
        gl.glDisable(self.heatmap.target)

    def output_loop(self, _):
        if self.frame % self.subframes == 0:
            self.control_loop()
        alpha = float(self.frame % self.subframes) / float(self.subframes)
        for car in self.cars:
            self.anim_x[car] = (1 - alpha) * self.prev_x[car] + alpha * car.x
        self.frame += 1

    def animation_loop(self, _):
        t = time.time()
        alpha = min((t - self.prev_t) / self.dt, 1.)
        for car in self.cars:
            self.anim_x[car] = (1 - alpha) * self.prev_x[car] + alpha * car.x

    def draw_lane_surface(self, lane):
        gl.glColor3f(0.4, 0.4, 0.4)
        W = 1000
        graphics.draw(4, gl.GL_QUAD_STRIP, ('v2f',
                                            np.hstack([lane.p - lane.m * W - 0.5 * lane.w * lane.n,
                                                       lane.p - lane.m * W + 0.5 * lane.w * lane.n,
                                                       lane.q + lane.m * W - 0.5 * lane.w * lane.n,
                                                       lane.q + lane.m * W + 0.5 * lane.w * lane.n])
                                            ))

    def draw_lane_lines(self, lane):
        gl.glColor3f(1., 1., 1.)
        W = 1000
        graphics.draw(4, gl.GL_LINES, ('v2f',
                                       np.hstack([lane.p - lane.m * W - 0.5 * lane.w * lane.n,
                                                  lane.p + lane.m * W - 0.5 * lane.w * lane.n,
                                                  lane.p - lane.m * W + 0.5 * lane.w * lane.n,
                                                  lane.p + lane.m * W + 0.5 * lane.w * lane.n])
                                       ))

    def draw_car(self, x, color='yellow', opacity=255):
        sprite = self.sprites[color]
        sprite.x, sprite.y = x[0], x[1]
        sprite.rotation = -x[2] * 180. / math.pi
        sprite.opacity = opacity
        sprite.draw()

    def draw_object(self, obj):
        sprite = self.obj_sprites[obj.name]
        sprite.x, sprite.y = obj.x[0], obj.x[1]
        sprite.rotation = obj.x[2] if len(obj.x) >= 3 else 0.
        sprite.draw()

    def on_draw(self):
        self.window.clear()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        self.camera()
        gl.glEnable(self.grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self.grass.target, self.grass.id)
        W = 10000.
        graphics.draw(4, gl.GL_QUADS,
                      ('v2f', (-W, -W, W, -W, W, W, -W, W)),
                      ('t2f', (0., 0., W * 5., 0., W * 5., W * 5., 0., W * 5.))
                      )
        gl.glDisable(self.grass.target)
        for lane in self.lanes:
            self.draw_lane_surface(lane)
        for lane in self.lanes:
            self.draw_lane_lines(lane)
        for obj in self.objects:
            self.draw_object(obj)
        for car in self.cars:
            if car != self.main_car and car not in self.visible_cars:
                self.draw_car(self.anim_x[car], car.color)
        if self.heat is not None:
            self.draw_heatmap()
        for car in self.cars:
            if car == self.main_car or car in self.visible_cars:
                self.draw_car(self.anim_x[car], car.color)
        gl.glPopMatrix()
        if isinstance(self.main_car, Car):
            self.label.text = 'Speed: %.2f' % self.anim_x[self.main_car][3]
            self.label.draw()
        if self.output is not None:
            pyglet.image.get_buffer_manager().get_color_buffer().save(self.output.format(self.frame))

    def reset(self):
        for car in self.cars:
            car.reset()
        self.prev_t = time.time()
        for car in self.cars:
            self.prev_x[car] = car.x
            self.anim_x[car] = car.x
        self.paused = False
        self.history_x = [[] for car in self.cars]
        self.history_u = [[] for car in self.cars]

    def run(self, filename=None, pause_every=None):
        self.pause_every = pause_every
        self.reset()
        if filename is not None:
            with open(filename) as f:
                self.feed_u, self.feed_x = pickle.load(f)
        if self.output is None:
            pyglet.clock.schedule_interval(self.animation_loop, 0.02)
            pyglet.clock.schedule_interval(self.control_loop, self.dt)
        else:
            self.paused = False
            self.subframes = 6
            self.frame = 0
            self.autoquit = True
            pyglet.clock.schedule(self.output_loop)
        self.event_loop.run()

    def run_modified(self, history_x, history_u):
        self.pause_every = None
        self.reset()
        self.feed_x = history_x
        self.feed_u = history_u
        pyglet.clock.schedule_interval(self.animation_loop, 0.02)
        pyglet.clock.schedule_interval(self.control_loop, self.dt)
        self.event_loop.run()

# These two functions help to plot trajectories at the same time.
def vis_traj(sim_obj, input_vals, name="unnamed", loc=None):
    sim_obj.feed(input_vals)
    sim_obj.watch(1, name=name, loc=loc)

def plot_for_ever(file_name, sim_obj, nuvec=10, name="unnamed", loc=None):
    while True:
        if os.path.exists(file_name):
            with open(file_name) as f:
                uvec = [float(x) for x in f.read().split("\n")[:nuvec]]
                f.close()
                os.remove(file_name)
                vis_traj(sim_obj, uvec, name=name, loc=loc)
        time.sleep(2)
