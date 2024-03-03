from flask import Flask, render_template, jsonify, request
from VizPlotting import VizPlotting

app = Flask(__name__)

Plot_obj = VizPlotting()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/contributors')
def contributors():
    return render_template('contributors.html')

@app.route('/get_intro_bar', methods=['GET','POST'])
def get_intro_bar():
    try:
        graphJSON = Plot_obj.create_intro_bargraph()
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 

    
@app.route('/get_pca_2d', methods=['GET','POST'])
def get_pca_2d():
    try:
        graphJSON = Plot_obj.create_pca_2d()
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 
    

@app.route('/get_pca_3d', methods=['GET','POST'])
def get_pca_3d():
    try:
        graphJSON = Plot_obj.create_pca_3d()
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 


@app.route('/get_tsne', methods=['GET','POST'])
def get_tsne():
    try:
        graphJSON = Plot_obj.create_tsne()
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 


@app.route('/get_umap', methods=['GET','POST'])
def get_umap():
    try:
        graphJSON = Plot_obj.create_umap()
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 


@app.route('/get_lle', methods=['GET','POST'])
def get_lle():
    try:
        graphJSON = Plot_obj.create_lle()
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 


@app.route('/get_svd', methods=['GET','POST'])
def get_svd():
    try:
        graphJSON = Plot_obj.create_svd()
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 

@app.route('/get_categorical_plot', methods=['GET','POST'])
def get_categorical_plot():
    try:
        category = request.args.get('category')
        graphJSON = Plot_obj.create_categorical_plot(category)
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 
    
@app.route('/get_numerical_plot', methods=['GET','POST'])
def get_numerical_plot():
    try:
        x_cat = request.args.get('x_cat')
        y_cat = request.args.get('y_cat')
        graphJSON = Plot_obj.create_numerical_plot(x_cat, y_cat)
        return jsonify(graphJSON)

    except Exception as e:
        return jsonify(error=str(e)), 500 

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
