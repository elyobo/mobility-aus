import os
import shutil
import time
import random
import zipfile
import glob
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.common import exceptions
from PIL import Image as PILImage

MAXWAIT = 60.
TIMEOUT = 60

def list_elements(element):
#     return element.find_elements_by_xpath('//*[@id]')
    return element.find_elements_by_css_selector("*")

def screen(target, name = 'screen'):
    filename = f"{name}.png"
    if hasattr(target, 'save_screenshot'):
        target.save_screenshot(filename)
    else:
        target.screenshot(filename)
    return PILImage.open(filename)

def get_element_pos(element):
    x, y, width, height = element.rect.values()
    return int(x + width / 2), int(y + height / 2)

def go_click(driver, refelement, coords, tocoords = None, click = True):
    if tocoords is None:
        xoffset, yoffset = coords
    else:
        xoffset, yoffset = (a - b for a, b in zip(tocoords, coords))
    action = webdriver.common.action_chains.ActionChains(driver)
    action.move_to_element(refelement)
    action.move_by_offset(xoffset, yoffset)
    if click:
        action.click()
    action.perform()
    random_sleep(1.)

def screen(target, name = 'screen'):
    filename = f"{name}.png"
    if hasattr(target, 'save_screenshot'):
        target.save_screenshot(filename)
    else:
        target.screenshot(filename)
    return PILImage.open(filename)

def get_element_startswith(element, text):
    for i, elem in enumerate(list_elements(element)):
        if not elem.text:
            continue
        if elem.text.startswith(text):
            return elem

def login(driver, loginURL, loginName, loginPass):
    print("Navigating to login page...")
    try:
        driver.get(loginURL)
    except exceptions.WebDriverException:
        raise ValueError("No login page found!")
    print("Navigated to login page.")

    print("Logging in...")

    random_sleep(0.5)
    username = driver.find_element_by_id("email")
    password = driver.find_element_by_id("pass")
    username.send_keys(loginName)
    random_sleep(0.2)
    password.send_keys(loginPass)
    random_sleep(0.2)
    try:
        submit = driver.find_element_by_id("loginbutton")
        submit.click()
    except exceptions.NoSuchElementException:
        try:
            submit = driver.find_element_by_name('login')
            submit.click()
        except exceptions.NoSuchElementException:
            password.send_keys(u'\ue007')

    random_sleep(0.5)
    try:
        _ = driver.find_element_by_id("loginbutton")
        raise Exception("Login failed!")
    except exceptions.NoSuchElementException:
        try:
            _ = driver.find_element_by_id("login_form")
            raise Exception("Login failed!")
        except exceptions.NoSuchElementException:
            pass
    print("Logged in.")

def go_to_datapage(driver, dataURL, searchstr):
    driver.get(dataURL)
    random_sleep(0.5)
    target = driver.find_element_by_id("js_1")
    target.click()
    random_sleep(0.5)
    target.send_keys(searchstr)
    random_sleep(0.5)
    return target

def random_sleep(factor = 1.):
    sleepTime = (random.random() + 1.) * factor
    time.sleep(sleepTime)
    return sleepTime

def wait_check(
        condition,
        message = None,
        repeatAction = None,
        repeatException = True,
        waitInterval = 1.,
        maxWait = None
        ):
    waited = 0.
    while not condition():
        if not message is None:
            print(message)
        waitTime = random_sleep(waitInterval)
        waited += waitTime
        if not maxWait is None:
            if waited > maxWait:
                raise Exception("Wait time exceeded!")
        if not repeatAction is None:
            if repeatException:
                repeatAction()
            else:
                try:
                    repeatAction()
                except:
                    pass


# from PIL import Image as PILImage
# im = PILImage.open('screen.png')
# im.crop((
#     210, 200, # mins
#     310, 250, # maxs
#     ))

class Driver:
    def __init__(self, options, profile, logDir = '.'):
        self.options, self.profile, self.logDir = options, profile, logDir
    def __enter__(self):
        global TIMEOUT
        self.driver = webdriver.Firefox(
            options = self.options,
            firefox_profile = self.profile
            )
        self.driver.set_page_load_timeout(TIMEOUT)
        return self.driver
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value:
            print("An error has occurred. Saving screenshot to target directory and exiting.")
            try:
                errorFilePath = os.path.join(self.logDir, str(int(time.time())) + '.png')
                self.driver.save_screenshot(errorFilePath)
            except:
                print("Another error occurred: could not save screenshot.")
        self.driver.quit()
        if os.path.isfile('geckodriver.log'):
            os.remove('geckodriver.log')
        if exc_type is not None:
            return False
        else:
            return True

class TempDir:
    def __init__(self, path, maxWait = None):
        self.path = path
        self.maxWait = maxWait
    def __enter__(self):
        os.makedirs(self.path, exist_ok = False)
        wait_check(lambda: os.path.isdir(self.path), maxWait = self.maxWait)
    def __exit__(self, exc_type, exc_value, exc_traceback):
        shutil.rmtree(self.path, ignore_errors = True)
        wait_check(lambda: not os.path.isdir(self.path), maxWait = self.maxWait)
        if exc_type is not None:
            return False
        else:
            return True

FBSEARCHSTRS = {
    '786740296523925': 'Melbourne coronavirus disease prevention',
    '1391268455227059': 'Victoria State coronavirus disease prevention',
    '1527157520300850': 'Sydney coronavirus disease prevention',
    '2622370339962564': 'New South Wales coronavirus disease prevention',
    }

def pull_data(
        fbid,
        loginName,
        loginPass,
        maxWait = MAXWAIT
        ):

    print(f"Downloading for {fbid}...")

    loginURL = 'https://www.facebook.com'
    dataURL = 'https://partners.facebook.com/data_for_good/data/?partner_id=467378274536608'
    dataMime = 'application/zip' #'text/csv'
    searchstr = FBSEARCHSTRS[fbid]
    targettext = 'Movement between tiles'

    repoPath = os.path.abspath(os.path.dirname(__file__))
    outDir = os.path.join(repoPath, 'data', fbid)
    os.makedirs(outDir, exist_ok = True)

    downloadDir = os.path.join(outDir, '_temp')

#     os.makedirs(downloadDir, exist_ok = True)
    with TempDir(downloadDir, maxWait = maxWait):

        profile = webdriver.FirefoxProfile()
        profile.set_preference("browser.download.folderList", 2)
        profile.set_preference("browser.download.manager.showWhenStarting", False)
        profile.set_preference("browser.download.dir", downloadDir)
        profile.set_preference("browser.helperApps.neverAsk.saveToDisk", dataMime)
        options = Options()
        options.add_argument("--headless")

        with Driver(options, profile, outDir) as driver:

            login(driver, loginURL, loginName, loginPass)

            searchbar = go_to_datapage(driver, dataURL, searchstr)

            target = get_element_startswith(driver, targettext)
            x, y = get_element_pos(target)

            go_click(driver, target, (250 - x, 40))

            go_click(driver, target, (900 - x, 450 - y))

            getzips = lambda: glob.glob(os.path.join(downloadDir, '*.zip'))
            def wait_condition():
                return len(getzips()) and not len(glob.glob(os.path.join(downloadDir, '*.part')))
            try:
                wait_check(wait_condition, maxWait = maxWait)
            except:
                driver.save_screenshot(os.path.join(outDir, 'screen.png'))
                raise Exception(f"Download failed for {fbid}; screenshot saved.")
            zipfilenames = getzips()

        random_sleep(3.)

        csvfilenames = glob.glob(os.path.join(outDir, '*.csv'))
        for zipfilename in zipfilenames:
            def open_zipfile():
                return zipfile.ZipFile(os.path.join(downloadDir, zipfilename), 'r')
            def wait_condition():
                try:
                    with open_zipfile() as _:
                        ...
                    return True
                except:
                    return False
            wait_check(wait_condition, maxWait = maxWait)
            with open_zipfile() as zfile:
                for zname in zfile.namelist():
                    if zname in csvfilenames:
                        print(f"File {zname} already acquired: skipping.")
                    else:
                        zfile.extract(zname, outDir)
                        print(f"Downloaded new for {fbid}: {zname}")

    print(f"Downloaded all for {fbid}.")

def pull_datas(fbids, *args, **kwargs):
    for fbid in fbids:
        try:
            pull_data(fbid, *args, **kwargs)
        except Exception as exc:
            print(f"Exception in {fbid}: {exc}")
