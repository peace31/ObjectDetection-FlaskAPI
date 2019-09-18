from flask import *
from functools import wraps
import os
import base64
import time
from scipy import misc
from flask import send_from_directory
import similar_category as sc
UPLOAD_FOLDER = os.getcwd() +'/uploads'
TRAIN_FOLDER = os.getcwd() +'/train'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# python flask api
app=Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRAIN_FOLDER'] = TRAIN_FOLDER

app.secret_key = 'mine'
#app.secret_key='fx3H2uRjJo3Xv/hiMS0HhrsYtGQeRrChqsL2Yosv'
# app.access_key='AKIAJYW5WVTLI7BCYCBQ'
# @app.route('/')
# def home():
#     return render_template('home.html')
# log in request
def login_required(test):
	@wraps(test)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			return test(*args, **kwargs)
		else:
			flash('You need to login first.')
			return redirect(url_for('log'))
	return wrap


# second page (main page)
@app.route('/main')
@login_required
def mainpage():
    error = None
    return render_template('home.html', error=error)
# check whether file is in allowed images or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# generate unique file name for uploaded image
def encode(string):
    filename, file_extension = os.path.splitext(string)
    timestamp = str(time.time()).replace('.','_');
    return filename + '_' + timestamp + file_extension
# return uploaded file full path
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
# upload evenet
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def imageupload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return json.dumps({'status': 'No file part'})
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return json.dumps({'status': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = encode(file.filename)
            path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            app.config['FileName'] = filename
            file.save(path)
            return json.dumps({'status':'ok', 'filename': filename})
    return json.dumps({'status': app.config['UPLOAD_FOLDER']})
# image crop function
def crop(x1,x2,y1,y2):
    path=os.path.join(app.config['UPLOAD_FOLDER'], app.config['FileName'])
    img=misc.imread(path)
    height, width, layers = img.shape
    w1=1170.0;h1=780.0;
    scale_x=float(width)/w1;scale_y=float(height)/h1
    mx1=int(int(x1)*scale_x);mx2=int(int(x2)*scale_x);my1=int(int(y1)*scale_y);my2=int(int(y2)*scale_y)
    Cimage=img[my1:my2,mx1:mx2]
    return Cimage
# find similar categories
def similar_category(Croped_image):
    d=os.listdir(app.config['TRAIN_FOLDER'])
    D=[]
    if(d.__len__()==0):
        return D
    elif(d.__len__()<=10):
        for i in range(d.__len__()):
            D.append(d[i])
        return D
    else:
        D=sc.s_category(Croped_image,app.config['TRAIN_FOLDER'])
        return D

# def get_category():
#     d=os.listdir(app.config['TRAIN_FOLDER'])
#     return d

@app.route('/crop', methods=['POST'])
@login_required
def imagecrop():
    error = None
    if request.method == 'POST':
        x1 = request.form['x1']
        y1 = request.form['y1']
        x2 = request.form['x2']
        y2 = request.form['y2']
        Croped_image=crop(x1,x2,y1,y2)
        listitem=similar_category(Croped_image)
        app.config['Crop_image']=Croped_image
        app.config['listitem'] = listitem
        #listitem = get_category()
        # listitem=[]
        return json.dumps({'status':'OK','category':listitem})
    else:
        return None
def save_image(category):
    d = os.listdir(app.config['TRAIN_FOLDER'])
    flag=0
    if(d.__len__()>0):
        for i in range(d.__len__()):
            if(d[i]==category):
                path=os.path.join(app.config['TRAIN_FOLDER'],d[i])
                path=os.path.join(path,app.config['FileName'])
                misc.imsave(path,app.config['Crop_image'])
                flag=1
                break
    if(flag==0):
        path = os.path.join(app.config['TRAIN_FOLDER'], category)
        os.mkdir(path)
        path = os.path.join(path, app.config['FileName'])
        misc.imsave(path, app.config['Crop_image'])

# save button event
@app.route('/save', methods=['POST'])
@login_required
def imagesave():
    error = None
    if request.method == 'POST':
        categories = json.loads( request.form['category'] )
        save_image(categories)
        listitem = app.config['listitem']
        return json.dumps({'status':'OK','category':listitem})
    else:
        return None
# log in page
@app.route('/', methods=['GET', 'POST'])
def log():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            session['logged_in'] = True
            return redirect(url_for('mainpage'))
    return render_template('log.html', error=error)


if __name__=='__main__':
    app.run(debug=True)
