import pandas as pd
from scipy.spatial.distance import cosine
from sqlalchemy import create_engine, Table, MetaData
from flask import Flask, jsonify, make_response, request


# Setup
app = Flask(__name__, static_url_path='') # For the API
engine = create_engine('postgresql://postgres:Morgen1995@localhost:5432/testdb') # Подключение к БД
source_table_name = 'source_data' # Сырые данные для обучающей выборки, которые находятся в CSV
band_sim_table_name = 'band_sim_matrix' # Матриця похожести, которая вычисляется из CSV
band_table_name = 'band_recs' # Таблица рекомендаций клиентов, которые мы храним

N_SIMILAR_BANDS = 10 # Находим 10 похожих исполнителей (константа

def init():
    '''
        Считывание таблиц в память и загрузка их в БД
    '''
    global insert_user_stmt, source_table, source_df, band_rec_df, band_similarity_matrix
    meta = MetaData()
    try:
        source_df = pd.read_sql_table(source_table_name, engine, index_col='index')
    except ValueError as e:
        print('Source table missing. Loading from CSV, and writing to DB.')
        source_df = pd.read_csv('data.csv')
        source_df.columns = [c.lower() for c in source_df.columns]
        source_df.to_sql(source_table_name, engine)
        print('Wrote source data to DB.')
    source_table = Table(source_table_name, meta, autoload=True, autoload_with=engine)
    insert_user_stmt = source_table.insert().values()
    try:
        band_similarity_matrix = pd.read_sql_table(band_sim_table_name, engine, index_col='index')
        band_rec_df = pd.read_sql_table(band_table_name, engine, index_col='index')
    except ValueError as e:
        print(e)
        calc_recs()

def write_df_to_db():
    print('Writing to db')
    band_similarity_matrix.to_sql(band_sim_table_name, engine, if_exists='replace')
    band_rec_df.to_sql(band_table_name, engine, if_exists='replace')
    return 'Wrote dataframe to db'

def getScore(history, similarities):
   return sum(history * similarities) / sum(similarities)

@app.route('/api/v1.0/recalc')
def calc_recs():
    '''
        Метод расчета рекомендаций в таблицу
        1) Таблица источник (source) используется для расчета матрицы схожести (SVD-матрицы). Это позволит нам рекомендовать
        похожих исполнителей на основе вкусов клиента. Например, клиенту, который слушает Metallica, будет рекомендовано
        послушать Iron Maiden.
        2) Таблица источник и таблица с рекомендациями используется, чтобы найти исполнителей, которые удовлетворяют вкусы
        клиента.
    '''
    global source_df, band_rec_df, band_similarity_matrix
    # Step 1) Коллаборативная фильтрации на основе предметов
    print('Calculating band similarities')

    data_bands = source_df.drop('user', 1)
    band_similarity_matrix = pd.DataFrame(index=data_bands.columns, columns=data_bands.columns)


    for i in range(0, len(band_similarity_matrix.columns)):
        # Проходим по каждому исполнителю
        print('New band starting, index: ' + str(i))
        for j in range(0, len(band_similarity_matrix.columns)):
          # Находим схожесть
          band_similarity_matrix.iloc[i, j] = 1 - cosine(data_bands.iloc[:,i], data_bands.iloc[:,j])
    print('Done loop for building cosine similarities')
    band_rec_df = pd.DataFrame(index=band_similarity_matrix.columns, columns=range(1, N_SIMILAR_BANDS + 1))
    for i in range(0, len(band_similarity_matrix.columns)):
       band_rec_df.ix[i, :N_SIMILAR_BANDS] = band_similarity_matrix.ix[0:, i].sort_values(ascending=False)[:N_SIMILAR_BANDS].index
    print('Done All! Inserting into DB')
    # Процесс расчета схожести закончен
    return write_df_to_db()

def get_rec_for_user(idx):
    '''
        Расчет каждого исполнителя для рекомендаций юзера. Возвращается таблица коеффициентов по каждому
        исполнителю
    '''
    print('Calculating user similarities')

    data_sims = pd.Series(index=source_df.columns)
    data_bands = source_df.drop('user', 1)

    i = idx
    for j in range(1, len(data_sims.index)):
        product = data_sims.index[j]

        if source_df.ix[i][j] == 1:
            # Юзер уже слышал песню, не будем ее рекомендовать
            data_sims.ix[j] = 0
        else:
            # Нашли исполнителей, на основе вкусов
            product_top_names = band_rec_df.ix[product][1:N_SIMILAR_BANDS]
            # Размещаем рекомендации в нисходящем порядке, от более похожего до менее
            product_top_sims = band_similarity_matrix.ix[product].sort_values(ascending=False)[1:N_SIMILAR_BANDS]
            # Получая массив песен для юзера
            user_purchases = data_bands.ix[idx, product_top_names]

            # Оценка расчитывается так
            # Есть 10 исполнителей, похожих на текущую
            # Из 10 исполнителей, юзеру нарвится 10
            # Использую эти знания и меру схожести для вывода топ-исполнителей
            data_sims.ix[j] = getScore(user_purchases, product_top_sims)

    return data_sims.sort_values(ascending=False)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.errorhandler(404)
def not_found(error):
    print(request.path)
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/api/v1.0/band/')
def list_bands():
    return jsonify({
        'bands': list(source_df.columns[1:]) # Skipping the user column
    })

@app.route('/api/v1.0/band/<path:name>')
def rec_band(name):
    '''
        JSON ответ со всеми исполнителями
    '''
    print(name in band_rec_df.index)
    if name in band_rec_df.index:
        # return jsonify(a=list(band_rec_df[band_rec_df['index'] == 'abba']))
        similar = list(
            band_rec_df.loc[name].ix[1:]
        )
    else:
        return make_response(
            jsonify({
                'error': 'Band {} not found'.format(name)
            }),
            404
        )

    resp_dict = {
        'name': name,
        'similar': similar
    }
    return jsonify(resp_dict)

@app.route()
def rec_user(id):
    '''
        Показать исполнителей, которые нравятся клиенту
    '''
    try:
        limit = int(request.args.get('limit'))
    except:
        limit = 10
    if id not in source_df.index:
        return make_response(
            jsonify({
                'error': 'User {} not found'.format(id)
            }),
            404
        )
    recs = list(get_rec_for_user(id).head(limit).index)
    like_list = list(source_df.iloc[id, :].loc[source_df.iloc[id, :] == 1].index)
    resp_dict = {
        'user': id,
        'likes': like_list,
        'recommendations': recs
    }
    return jsonify(resp_dict)

@app.route('/api/v1.0/user/', methods=['GET'])
def list_users():
    return jsonify({
        'num_of_users': source_df.shape[0]
    })

@app.route('/api/v1.0/user/', methods=['POST'])
def add_user():
    global source_df
    data = request.get_json(force=True)
    if not data or not data['likes']:
        return jsonify({
            'error': 'Need a JSON request formatted like: { "likes": ["simple plan", "abba", "coldplay"] }'
        })
    elif all(band_name in source_df.columns for band_name in data['likes']):
        print(data)
        user_data = dict.fromkeys(source_df.columns, 0)
        for band_name in data['likes']:
            if band_name != 'index' and band_name != 'user':
                user_data[band_name] = 1
        # Add to pandas DF here
        source_df = source_df.append(user_data, ignore_index=True)
        user_data['index'] = source_df.shape[0] - 1 # Subtract 1 since we *just* added an index
        # Add to db here
        
        conn = engine.connect()
        conn.execute(insert_user_stmt, **user_data)
        conn.close()

        return jsonify({
            'user_id': user_data['index']
        })
    else:
        return jsonify({
            'error': 'One or more bands were not found in the list.'
        })

if __name__ == '__main__':
    init()
    app.run(debug=True)
