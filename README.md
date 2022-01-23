## Segmentation pipelines

Репозиторий для обучения сегментационных сетей и их конвертации
в TensorRT.

## Основные зависимости
* [Pytorch](https://pytorch.org/get-started/previous-versions/) (я использовал 1.5)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) - модели
из этого репозитория могут быть сконвертированы в TensorRT. При использовании моделей из других
репозиториев подобное утверждение может быть неверным.
* [Albumentations](https://github.com/albumentations-team/albumentations) для 
аугментаций данных.
* [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
* Остальное перечислено в requirements.txt

## Код

Все аргументы интерфейса командной строки расположены в файле 
[arguments.py](src/arguments.py).

### Обучение
Обучение осуществляется посредством скрипта `train.py`.
Параметры CLI:
* `--data_root`: Путь к датасету.
* `--num_classes`: Количество семантических классов.
* `--model_name`: Имя, под которым будут сохраняться чекпоинты нейросети.
* `--model`: Путь к чекпоинту, с которого нужно продолжить обучение. 
Если указывается несуществующий файл, то ничего не подгружается, а 
обучение всё равно запускается.
* `--mode`: Режим обучения. Если не знаете, как быть, оставляйте значение 
по умолчанию (`seg`). Другие режимы можно посмотреть 
в [arguments.py](src/arguments.py). В гольфе они задействованы не были.
* `--use_mixup`: Mix-up аугментация. Не испульзуйте, если не уверены.
В гольфе задействована не была.
* `--backend`: Backbone нейронки, у нас по умолчанию используется
`efficientnet-b0`. Другие доступные можно смотреть 
[здесь](https://github.com/qubvel/segmentation_models.pytorch/tree/v0.2.0#encoders-).

#### Структура датасета
Здесь [`data_root`](README.md#14) - это параметр из CLI скрипта для обучения
```
data_root:
    +--images:
        +--1.png
        +--abc.png
        +--...
    +--indexes:
        +--1_color_mask.png
        +--abc_color_mask.png
        +--...
```
Если есть необходимость в другом формате датасета или как-то 
скомпоновать классы, можно в качестве примера посмотреть на [`dirt_dataset.py`](src/data/dirt_dataset.py).
Чтобы лучше понять, можно почитать официальный 
[гайд](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

### Визулизация масок
Для визуализации работы сети можно воспользовать скриптом
[`color_mask.py`](src/color_mask.py)
Параметры CLI:
* `--images_path`: Путь к папке с изображениями или одному конкретному изображению.
* `--num_classes`: Количество семантических классов.
* `--model_name`: Путь к pth файлу с весами модели.
* `--output_dir`: Папка, в которой будут сохранены визуализации.
* `--size`: Ширина и высота, с которыми изображения будут подаваться в сеть.
* `--colors`: Путь к csv файлу с картой цветов (пример: [colors.csv](src/colors.csv)).

## Конвертация
Общая схема такова: Pytorch -> ONNX -> TensorRT

### Конвертация в ONNX
Конвертация проходит из pth модели, в этом формате
сохраняет модели pytorch, который используется для обучения.
Запускается конвертация примерно следующим образом:

`python3 convert_to_onnx.py
--model_in {path-to-pth-model}.pth
--model_out {output-model-path}.onnx
--num_classes 11
--size (1280, 1344)
--backend efficientnet-b0`

Почитать, что эти аргументы значат можно командой 
`python3 convert_to_onnx.py --help`

По ширине и высоте входных данных, единственное ограничение - делимость
на 64.

### Конвертация в TensorRT
Конвертация проходит из onnx формата и осуществляется следующей 
командой: 

`trtexec --onnx={name-of-the-input-model}.onnx 
--shapes=input_1:1x640x1280x3 --explicitBatch --workspace=2048 
--saveEngine={name-of-the-output-model}.bin 
--fp16`

#### Некоторое описание параметров
Подробное описание можно получить командой `trtexec --help`.
* `--onnx`: Путь к модели в формате .onnx
* `--shapes`: Название и размерность входного тензора. 
Формат: name:{batch_size}x{height}x{width}x{channels}. 
В данном репозитории для входного тензора используется имя 'input_1'
* `--workspace`: Размер (в Мб) рабочего пространства во время конвертации 
и подбора стратегий. Не влияет на размер необходимой памяти во 
время работы сети.
* `--saveEngine`: Путь к выходной модели в формате .bin. 
* `--fp16`: Указание использовать точность fp16

Конвертация опробована в версии TensorRT 7.1, в частности 7.1.3. 