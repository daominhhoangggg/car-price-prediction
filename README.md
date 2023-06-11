# Xây dựng mô hình hồi quy tuyến tính và ứng dụng vào dự đoán giá xe hơi
Dự án này là một ứng dụng web để dự báo giá xe hơi sử dụng mô hình hồi quy. Ứng dụng cho phép người dùng nhập các biến đầu vào và trả về dự báo về giá xe hơi.

## Nội dung

1. [Thông tin dự án](#thông-tin-dự-án)
2. [Cài đặt](#cài-đặt)
3. [Sử dụng](#sử-dụng)
4. [Dữ liệu](#dữ-liệu)
5. [Cấu trúc dự án](#cấu-trúc-dự-án)

<a name="thông-tin-dự-án"></a>
## Thông tin dự án

- app.py: File Python chứa mã nguồn cho ứng dụng web. Nó sử dụng mô hình hồi quy để đưa ra dự báo giá xe hơi và hiển thị trên trang web giả lập chạy trên localhost.
- build_model.ipynb: File Jupyter Notebook chứa quá trình xây dựng mô hình và phân tích dữ liệu.
- index.html: File HTML để người dùng nhập các biến đầu vào.
- result.html: File HTML để hiển thị kết quả dự báo khi người dùng đã nhập đầy đủ các thuộc tính.
- /data: Thư mục chứa dữ liệu về xe hơi, bao gồm các trường như năm sản xuất, dung tích động cơ, số dặm đã đi và giá là biến phụ thuộc.
- /templates: Thư mục chứa các file HTML templates.
- /static/images: Thư mục chứa các hình ảnh sử dụng trong trang web.
- report.pdf: File PDF chứa báo cáo chi tiết về dự án.

<a name="cài-đặt"></a>
## Cài đặt
Để cài đặt dự án và chạy ứng dụng, làm theo các bước sau:

1. Clone repository này vào máy của bạn.
2. Chạy file app.py để khởi động ứng dụng.

<a name="sử-dụng"></a>
## Sử dụng

1. Mở trình duyệt web và truy cập vào địa chỉ http://localhost:5000.
2. Điền các biến đầu vào trong form và nhấn nút "Dự báo".
3. Kết quả dự báo sẽ được hiển thị trên trang web.

<a name="dữ-liệu"></a>
## Dữ liệu
Dữ liệu về xe hơi được lưu trong thư mục '/data'. Tập dữ liệu này bao gồm các trường như năm sản xuất, dung tích động cơ, số dặm đã đi và giá là biến phụ thuộc. Dữ liệu được sử dụng để xây dựng mô hình hồi quy.

<a name="cấu-trúc-dự-án"></a>
## Cấu trúc dự án
Cấu trúc thư mục của dự án như sau:

- app.py
- build_model.ipynb
- index.html
- result.html
- /data
    - audi.csv
    - bmw.csv
    - focus.csv
    - ford.csv
    - hyundi.csv
    - merc.csv
    - skoda.csv
    - toyota.csv
    - vauxhall.csv
    - vw.csv
- /templates
    - index.html
    - result.html
- /static/images
    - anh-chu-thich.png
    - residuals-ols.png
    - residuals-sklearn.png
- report.pdf
