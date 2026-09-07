# OCR PDF Tool

Tool này nhận PDF và xuất Markdown/Word, hỗ trợ PDF scan, PDF digital, trích xuất ảnh và xử lý theo lô.

## Cài đặt

Từ thư mục gốc repository:

```powershell
python -m pip install -r ocr/requirements.txt
```

## Chạy một PDF

```powershell
cd ocr
python main.py --input input/caythuoc.pdf --output output --mode auto
```

Các mode:

- `auto`: tự nhận diện PDF scan hay digital.
- `scan`: ép chạy OCR.
- `digital`: ép xử lý PDF digital.

## Chạy hàng loạt

Đặt các PDF cần xử lý vào `ocr/input/`, sau đó chạy:

```powershell
cd ocr
python main.py --input input --output output --batch
```

Kết quả được lưu trong `ocr/output/`. Thư mục input/output và artifact trung gian là local, không commit lên Git.

## Notebook

Notebook thử nghiệm nằm tại [`notebooks/main.ipynb`](notebooks/main.ipynb). Cấu hình mặc định nằm trong [`config.py`](config.py).
