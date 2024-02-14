# IIvMobS-2-LR2

## 🤡 Искусственный интеллект в мобильных системах, Часть 2 (семестр 4)

### 4. Фреймворк TensorFlow для обнаружения и локализации объектов на основе нейронных сетей

----------

## ⚠️ Если вы не jundevchik, пожалуйста, переключите ветку

Для этого, разверните список веток (там, где первая ветка называется main) и выберите свою

Список веток:

1. `main`
2. `f3rni`
3. `keepalove`
4. `katiklex`
5. `g0n4ar0v`
6. `drmoonshine`
7. `jundevchik` <

----------

## 🕒 Подготовка к запуску лабовы

> Приветствую, **jundevchik**! Для того, чтобы сделать эту лабову, нужно поставить Python. Установить Python на **Linux** зачастую можно, используя встроенный менеджер пакетов. Например, для дистрибутивов на основе **Arch Linux**: `pacman -S python`. Для **Windows**, скачать Python можно на официальном сайте: <https://www.python.org/downloads/>

1. Установите Python версии 3.9 или выше, если не установлено
   1. Проверить установку можно командой в терминале `python --version`
   2. Однако, **на Windows**, если вы устанавливали Python без автоматического добавления в системные переменные, вам необходимо найти путь к исполняемому файлу. Обычно это `C:\Users\Ваше имя пользователя\AppData\Local\Programs\Python\PythonВЕРСИЯ\python.exe`. **Если это так, в шаге 4 (ТОЛЬКО В ШАГЕ 4), вместо `python` вам необходимо использовать `"C:\Users\Ваше имя пользователя\AppData\Local\Programs\Python\PythonВЕРСИЯ\python.exe"`**
      1. В таком случае, чтобы проверить версию:

      ```shell
      "C:\Users\Ваше имя пользователя\AppData\Local\Programs\Python\PythonВЕРСИЯ\python.exe" --version
      ```

   3. **На Linux**, исполняемый файл Python обычно находится по пути `/usr/bin/python`. Узнать расположение можно, прописав `where python`

2. Установите консольную версию Git, если не установлено (или, скачайте вашу ветку напрямую с GitHub как архив)

3. Откройте терминал и клонируйте вашу ветку репозитория, используя команду (или, как было сказано ранее, скачайте эту ветку архивом)

   ```shell
   git clone -b jundevchik --single-branch https://github.com/F33RNI/IIvMobS-2-LR2
   ```

4. Откройте терминал и перейдите в директорию используя команду
   1. На **Linux**: `cd "путь/к папке IIvMobS-2-LR2"`
   2. На **Windows**: `cd "путь\к папке IIvMobS-2-LR2"`
5. Создайте виртуальную среду, используя команду `python -m venv venv`
6. Активируйте виртуальную среду. Для этого пропишите в терминале
   1. На **Linux**: `source venv/bin/activate`
   2. На **Windows**: `venv\Scripts\activate.bat`.
7. Если виртуальная среда создана и запущена верно, в терминале перед текущей рабочей директорией появится `(venv)` (или иное обозначение виртуальной среды, в зависимости от настроек вашего терминала)
8. Установите все пакеты для лабовы, используя команду `pip install -r requirements.txt`
9. Для проверки, пропишите `pip list`. Вы должны увидеть установленные пакеты. Среди них должно быть `opencv-python`, `label-studio` и `icrawler`
10. Готово!

----------

## 🏗️ Запуск лабовы и создание отчёта

> Несмотря на то, что лабораторная называется "Фреймворк TensorFlow для обнаружения и локализации объектов на основе нейронных сетей", и 2/3 лекции в ЛМС действительно отдалённо напоминают что-то про TensorFlow, ни в конце лекции, ни в выполненной работе, TensorFlow не используется 🤡
>
> Если делать **по заданию**, то в начале нужно при помощи модели от Caffe распознать хоть что-то на любом изображении. Потом нужно дообучить нейронку для распознавания объекта по варианту, затем сделать риал-тайм распознавание и потом зачем-то опять "дообучить нейронную сеть для распознавания в видео-потоке объектов согласно варианту в таблице выше" ¯\\\_\(ツ\)\_/¯
>
> На деле же, для распознавания, была взята модель YOLO v4 Tiny. В ней есть 80 классов (их можно посмотреть в файле `coco.names.txt`) и несколько из этих классов пересекаются с вариантами в задании. Поэтому ничего обучать не придётся. Также, благодаря использованию OpenCV, нет разницы в использовании изображения, видеофайла или веб-камеры. Поэтому для выполнения всех поставленных требований придётся немножко ~~законсперировать~~ подогнать лабу...
>
> Пример отчёта (и другой версии кода) можно найти в ветке `f3rni`. **Не копируйте отчёт. Ваш отчёт должен отличаться и быть уникальным. Иначе забракуют и вашу, и мою лабу. Не используйте представленные в отчёте текстовые конструкции, и, тем более, изображения!**
>
> **Дисклеймер:** _вы всё делаете на свой страх и риск. Авторка данного репозитория не несёт ответственности ни за какой моральный или физический ущерб, связанный с использованием кода, отчётов или иных файлов, расположенных в репозитории. Эти лабы распространяются под лицензией The Unlicensed. Это означает что все риски вы принимаете на себя (и можете делать с кодом всё что угодно)_.

1. Определитесь с вариантом. Он должен быть уникальным. Выбирать можно только из вариантов со свёздочкой. Скопируйте название объекта из колонки `Название из coco.names.txt`. Варианты можно найти в секции `🫱 Варианты`, в самом низу этого файла
2. 🤫 ~~Первый этап конспирации:~~ Найдите в файле `detection.py` код ниже, раскомментируйте его и вставьте туда название класса, скопированное в 1 пункте. Это уберёт рамочки и текст с обнаруженного класса, чтобы выглядело так, как будто нейросеть не знает такого класса

   ```python
   if classes[class_index] == "bear":
       continue
   ```

3. Найдите 2-3 изображения с разными объектами. Ваш объект тоже там может быть, но помимо него, важно чтобы было и что-то другое. Также найдите 1-2 изображения где точно есть ваш объект. Постарайтесь, чтобы на найденных изображениях не было предметов из других вариантов (самолёт, книга, часы, слон, жираф). Желательно, чтобы все изображения были в формате JPEG или PNG. В папке `photos` есть одна демо-картинка. Её можно также использовать
4. Вставьте путь к первому изображению в `cv2.VideoCapture()` и запустите скрипт, используя `python main.py`. Убедитесь, что на изображении распозналось хоть что-то верно, и, что ваш класс **не распознался**. Сделайте и сохраните скриншоты для дальнейшего использования в отчёте. Изменяйте `cv2.VideoCapture()` и сохраняйте скриншоты. Думаю, 1-3 скриншота для отчёта будет достаточно
5. В отчёте опишите процесс использования натренированной модели. Можете показать структуру директории `Resources`. Кратко приведите основные функции, отвечающие за загрузку и использование модели
6. Вставьте сохранённые ранее скриншоты и объясните, что ваш класс из варианта нейросеть не распознаёт
7. 🤫 ~~Второй этап конспирации:~~ Теперь нужно будет изобразить процесс создания датасета. Для этого в начале нужно скачать изображения, например из гугл картинок. По заданию нужно 100 изображений. Для этого, напишите в терминале:

   ```python
   python -c "from icrawler.builtin import GoogleImageCrawler; GoogleImageCrawler(storage={'root_dir': 'dataset'}).crawl(keyword='bear', max_num=100)"
   ```

   По окончанию скачивания, в папке dataset будет 100 скаченных изображений

8. Сделайте скриншот процесса скачивания / скаченных изображений для отчёта
9. Запустите label-studio, написав в терминале `label-studio`. Дождись загрузки, должен открыться браузер по-умолчанию. (Обычно адрес сервера: <http://localhost:8080/>)
10. Создайте новый аккаунт (если не существует). Для регистрации советую использовать сервис временных e-mail, например <https://temp-mail.org/>
11. Нажмите `Create Project`. В поле `Project Name` напишите что-нибудь что может доказать что это ваша лаба, например `LR2 jundevchik`
12. На вкладке `Labeling Setup` выберите `Object Detection with Bounding Boxes`
13. Нажмите `Save`
14. Нажмите `Go to import` и в открывшееся поле перетащите все 100 скаченных изображений в папке `dataset`. После загрузки нажмите `Import`
15. Сверху нажмите `Settings` -> `Labeling Interface`
16. Справа удалите все существующие классы. В поле `Add label names` впишите название вашего класса, скопированное в пункте 1. Нажмите `Add`. Ваш класс должен появиться в списке справа. Нажмите `Save`
17. Снова перейдите в проект и нажмите `Label All Tasks`
18. Должно открыться первое скаченное изображение. Нажмите снизу на ваш класс и выделите объект интереса в прямоугольник. Нажмите `Submit`. Повторите ещё для пары изображений **и как увидите удачное изображение, выделите в прямоугольник и сохраните скриншот для отчёта**. Лучше сохраните 2-3 скриншота, остальные изображения размечать не нужно (всё равно же обучать никто не будет 🙃)
19. ~~Как надоест~~ Как сделаете 2-3 качественных скриншота, нажмите на стрелочку `\/` (около кнопки `Submit`) и выберите `Submit and exit`
20. Снова нажмите на ваш проект и нажмите вверху `Export`. Выберите формат `YOLO` и сохраните ваш датасет куда-нибудь. Если хочется, можно тоже сделать скриншот экспорта / содержимого архива для отчёта
21. Можно закрывать окно label-studio и нажать `Ctrl` + `C` в терминале для остановки сервера label-studio
22. Далее, дополнительно, в отчёт можно вставить скриншот раздела загрузки сайта <https://cocodataset.org> и написать, что скаченный датасет был объединён с созданными и размеченными изображениями
23. 🤫 ~~А теперь, время настоящей-конспирации!~~ Вам понадобятся скриншоты из видосов по лабам по "Распознавание образов". Вставьте в отчёт скриншот процесса обучения 3-ей или 4-ой ЛР по "Распознавание образов". На скриншоте не должно быть понятно, какая именно модель обучается. Для большего эффекта, в отчёте можно написать про TensorBoard и привести какой-то скрин из тех же лабораторных "Распознавание образов". В общем, нужно создать видимость сложного и долгого процесса обучения (_оригинальный датасет так-то ~150Гб весит, если не больше_)
24. **Удалите** код, раскомменченный на этапе 2. И прогоните несколько изображений (также, как в пункте 4), на которых **присутствует** объект из варианта. Объект должен распознаться. Сделайте скриншоты и вставьте их в отчёт. Объясните, что благодаря дообучению модели, теперь нужные объекты распознаются
25. Теперь время реализовать распознавание видео-потока. Напишите в отчёте, что, якобы, ранее, для загрузки изображения из файла использовалась функция `cv2.imread()` (~~_на деле это не так_~~). А теперь, будем использовать класс `cv2.VideoCapture` (~~_который и так использовался ранее_~~). Опишите, что этот класс принимает на вход или пусть к видеофайлу или индекс веб-камеры и позволяет читать стрим покадрово. Опишите что каждый кадр затем обрабатывается как отдельное изображение и показывается пользователю при помощи функции `cv2.imshow()`
26. Найдите на просторах Интернета видео с вашим объектом интереса, или, если хотите, можете использовать веб-камеру, если дома у вас есть такой объект
27. Укажите путь к видеофайлу или индекс вебкамеры (начните с `0`, если не верная вебкамера то `1` и т.д.) в `cv2.VideoCapture()`. Например, если нужно использовать веб-камеру то: `cv2.VideoCapture(0)`
28. Запустите скрипт, прописав `python main.py`. Ваш объект должен будет распознаться. Начните запись видео (это одно из требований отчёта)
29. Далее из видео возьмите несколько кадров для отчёта, а сам видеофайл необходимо загрузить вместе с отчётом в ЛМС
30. Думаю, на этом всё. Внимательно проверяйте уникальность отчёта и законсперированность процесса обучения. Всё должно выглядеть уникально и реалистично. Когда будете вставлять код в приложение, не забудьте удалить шапку лицензии (чтобы не вызвало никаких подозрений у непонимающих) и код ~~конспирации~~ раскоменченный во втором пункте. Также, крайне советую удалить в коде докстринги у функций, если они имеются (большие комментарии, начинающиеся с `"""`)

🍀

----------

## 🤝 Участие в разработке

Если вы хотите что-то добавить / исправить ошибку, создайте форк своей ветки (`jundevchik`), внесите изменения и сделайте пул-реквест из своего форка в эту ветку (`ваш форк ветки jundevchik` -> `jundevchik`). Также, если считаете нужным, вы можете предложить изменения и для других веток

Пожалуйста, старайтесь придерживаться стандарта Conventional Commits при добавлении описания к коммитам. Больше информации по ссылке: <https://www.conventionalcommits.org/en/v1.0.0/>

----------

## 🫱 Варианты

Ниже указан номер варианта и объект для распознавания, для которого необходимо _"дообучить" нейронную сеть (вклад в оценку - 20%)_

Звёздочкой помечены варианты, классы которых уже существуют **_(брать только эти варианты)_**

```text
№ варианта  Объект      Название из coco.names.txt

    1.      Карандаш    -
    2.      Верблюд     -
    3.      Лягушка     -
*   4.      Медведь     bear
*   5.      Слон        elephant
    6.      Носорог     -
    7.      Бегемот     -
*   8.      Жираф       giraffe
*   9.      Зебра       zebra
    10.     Змея        -
    11.     Енот        -
    12.     Лось        -
    13.     Очки        -
*   14.     Книга       book
*   15.     Ножницы     scissors
*   16.     Часы        clock
*   17.     Самолет     aeroplane
    18.     Вертолет    -
    19.     Рыба        -
    20.     Пистолет    -
    21.     Танк        -
    22.     Оса         -
    23.     Голубь      -
*   24.     Корабль     boat
    25.     Заяц        -
```
