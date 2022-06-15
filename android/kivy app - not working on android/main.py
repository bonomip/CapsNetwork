from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import cv2
import numpy as np
import tensorflow as tf
from model_10_epochs import show_plain_output

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 640)
        play: False
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
    Label:
        id: label
        text_size: self.size
        halign: 'center'
        valign: 'middle'
        markup: True
''')


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        #convert camera snapshot to np array
        height, width = camera.texture.height, camera.texture.width
        #convert it
        newvalue = np.frombuffer(camera.texture.pixels, np.uint8)
        newvalue = newvalue.reshape(height, width, 4)
        gray = cv2.cvtColor(newvalue, cv2.COLOR_RGBA2GRAY)
        #trim image before resizing           
        h, w = gray.shape
        x = int( (w - h) / 2)
        gray = gray[0:h, x:(w-x)]
        #resize
        gray_scaled = cv2.resize(gray, (28, 28), interpolation = cv2.INTER_CUBIC) 
        #convert it
        image = np.array(gray_scaled)
        #save it
        #im = Image.fromarray(image)
        #im.save("test_scaled.jpeg")
        x = np.asarray(image)
        x = ( 255 - x ) / 255.0
        x = (x > 0.6) * x
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.reshape(x, [28, 28, 1])

        pred = show_plain_output(x)

        m = 0
        idx = 0
        
        for i in range(0,10):
            p = float(pred[0, 0, i, 0])
            if p > m:
                m = p
                idx = i

        s = ""
        for i in range(0, 10):
            p = np.around(float(pred[0, 0, i, 0]), decimals=2)
            if i == idx:
                 s += "[color=ff3333]Digit: "+str(i)+" - "+str(p)+"[/color]\n"
            else:
                s += "Digit: "+str(i)+" - "+str(p)+"\n"

        label = self.ids['label']
        label.text = s


class CapsNet(App):
    def build(self):
        return CameraClick()

if __name__ == "__main__":
    CapsNet().run()