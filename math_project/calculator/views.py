
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import io
import base64

from django.http import JsonResponse
from scipy.optimize import linprog
from django.shortcuts import render
import math
import re

def factorial(n):
    return math.factorial(n)

def basic_calculation(request):
    result = ''
    if request.method == 'POST':
        expression = request.POST.get('expression')

        # 处理组合数 C(m,n) 和 排列数 A(m,n)
        expression = re.sub(r'C\((\d+),\s*(\d+)\)', lambda m: str(math.comb(int(m.group(1)), int(m.group(2)))), expression)
        expression = re.sub(r'A\((\d+),\s*(\d+)\)', lambda m: str(math.perm(int(m.group(1)), int(m.group(2)))), expression)

        # 处理 log(m, n)
        expression = re.sub(r'log\((\d+),\s*(\d+)\)', lambda m: str(math.log(int(m.group(2)), int(m.group(1)))), expression)

        # 替换 e 和 π
        expression = expression.replace('e', str(math.e))
        expression = expression.replace('π', str(math.pi))

        # 替换三角函数和反三角函数
        expression = expression.replace('sin', 'math.sin')
        expression = expression.replace('cos', 'math.cos')
        expression = expression.replace('tan', 'math.tan')
        #expression = expression.replace('arcsin', 'math.asin')
        #expression = expression.replace('arccos', 'math.acos')
        #expression = expression.replace('arctan', 'math.atan')

        # 处理指数符号 ^ 和阶乘
        expression = expression.replace('^', '**')
        expression = re.sub(r'(\d+)\!', lambda m: str(math.factorial(int(m.group(1)))), expression)

        # 处理 ln
        expression = expression.replace('ln', 'math.log')

        # 处理平方根
        expression = re.sub(r'√\s*\(?', 'math.sqrt(', expression)

        # 计算其他表达式
        #   result = eval(expression)
        #except Exception as e:
        #    return JsonResponse({'result': '错误: ' + str(e)})

        try:
            result = eval(expression)
        except SyntaxError as e:
            return JsonResponse({'result': '语法错误: ' + str(e)})
        except NameError as e:
            return JsonResponse({'result': '未定义的名称: ' + str(e)})
        except Exception as e:
            return JsonResponse({'result': '错误: ' + str(e)})

        return JsonResponse({'result': result})

    return render(request, 'calculator/basic_calculation.html', {'result': result})

def linear_algebra(request):
    result = {}
    if request.method == 'POST':
        # 矩阵运算部分
        matrix_str = request.POST.get('matrix')
        operation = request.POST.get('operation')

        try:
            matrix = np.array(eval(matrix_str))

            if operation == 'add':
                matrix2_str = request.POST.get('matrix2')
                matrix2 = np.array(eval(matrix2_str))
                if matrix.shape != matrix2.shape:
                    result['error'] = '矩阵维数不一致，无法进行加法'
                else:
                    result['result'] = matrix + matrix2

            elif operation == 'subtract':
                matrix2_str = request.POST.get('matrix2')
                matrix2 = np.array(eval(matrix2_str))
                if matrix.shape != matrix2.shape:
                    result['error'] = '矩阵维数不一致，无法进行减法'
                else:
                    result['result'] = matrix - matrix2

            elif operation == 'multiply':
                matrix2_str = request.POST.get('matrix2')
                matrix2 = np.array(eval(matrix2_str))
                if matrix.shape[1] != matrix2.shape[0]:
                    result['error'] = '矩阵维数不一致，无法进行乘法'
                else:
                    result['result'] = np.dot(matrix, matrix2)

            elif operation == 'inverse':
                if matrix.shape[0] == matrix.shape[1] and np.linalg.det(matrix) != 0:
                    result['result'] = np.linalg.inv(matrix)
                else:
                    result['error'] = '该矩阵不可逆'

            elif operation == 'properties':
                result['shape'] = matrix.shape
                result['rank'] = np.linalg.matrix_rank(matrix)
                result['det'] = np.linalg.det(matrix) if matrix.shape[0] == matrix.shape[1] else None
                result['eigenvalues'] = np.linalg.eigvals(matrix)
                if matrix.shape[0] == matrix.shape[1]:
                    eigvals, eigvecs = np.linalg.eig(matrix)
                    result['eigenvectors'] = eigvecs

        except Exception as e:
            result['error'] = str(e)

        # 解线性方程组部分
        if request.POST.get('operation') == 'solve_system':
            matrix_str = request.POST.get('matrix')
            b_str = request.POST.get('b')

            try:
                A = np.array(eval(matrix_str))
                b = np.array(eval(b_str))

                if A.shape[0] != b.shape[0]:
                    result['error'] = '矩阵A的行数与向量b的维数不一致'
                else:
                    solution = np.linalg.solve(A, b)
                    result['solution'] = solution.tolist()

            except np.linalg.LinAlgError as e:
                result['solution'] = '无解或无穷解'
            except Exception as e:
                result['error'] = str(e)

        # 求解线性规划部分
        if request.POST.get('operation') == 'solve_lp':
            c_str = request.POST.get('c')
            A_str = request.POST.get('matrix')
            b_str = request.POST.get('b')
            optimization = request.POST.get('optimization')

            try:
                c = np.array(eval(c_str))
                A = np.array(eval(A_str))
                b = np.array(eval(b_str))

                if optimization == 'min':
                    res = linprog(c, A_ub=A, b_ub=b)
                else:
                    res = linprog(-c, A_ub=A, b_ub=b)

                if res.success:
                    result['lp_solution'] = res.x.tolist()
                    result['lp_status'] = '唯一最优解' if res.status == 0 else '可选择的最优解'
                else:
                    result['lp_status'] = '无解或无界'
            except Exception as e:
                result['error'] = str(e)

    return render(request, 'calculator/linear_algebra.html', {'result': result})


def simplex(request):
    # 单纯形法的实现可以在这里添加
    return render(request, 'calculator/simplex.html')

def calculus(request):
    result = {}
    graph = ''
    if request.method == 'POST':
        function_str = request.POST.get('function')
        x, y, z = sp.symbols('x y z')

        # 判断是否是方程或函数
        equal_signs = function_str.count('=')

        if equal_signs > 2:
            return render(request, 'calculator/calculus.html', {'result': result, 'graph': graph})

        # 分离方程和函数
        if equal_signs == 1:
            equation = function_str.split('=')
            lhs = sp.sympify(equation[0])
            rhs = sp.sympify(equation[1])
            implicit_derivative = sp.diff(lhs, x)  # 求隐函数的导数
            result['implicit_derivative'] = f"导数: {implicit_derivative} = {rhs}"
            function = lhs
        else:
            function = sp.sympify(function_str)

        # 一元函数处理
        if function.has(x) and not function.has(y) and not function.has(z):
            derivative = sp.diff(function, x)
            integral = sp.integrate(function, x)
            result['derivative'] = (derivative, sp.diff(function, x, 2))  # 一阶和二阶导
            result['integral'] = integral

            # 绘图
            f_lambdified = sp.lambdify(x, function, modules='numpy')
            x_vals = np.linspace(-10, 10, 400)
            y_vals = f_lambdified(x_vals)

            plt.figure()
            plt.plot(x_vals, y_vals, label=str(function))
            plt.title('函数图像')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.axhline(0, color='black', lw=0.5, ls='--')
            plt.axvline(0, color='black', lw=0.5, ls='--')
            plt.grid()
            plt.legend()
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)

        # 多元函数处理
        else:
            derivatives = {}
            if function.has(x):
                derivatives['x'] = (sp.diff(function, x), sp.diff(function, x, 2))
            if function.has(y):
                derivatives['y'] = (sp.diff(function, y), sp.diff(function, y, 2))
            if function.has(z):
                derivatives['z'] = (sp.diff(function, z), sp.diff(function, z, 2))
            result['derivative'] = derivatives

            # 绘图时只选定一个变量，比如x
            if function.has(x) and not function.has(y):
                f_lambdified = sp.lambdify(x, function.subs(y, 0), modules='numpy')  # 这里可以根据需要调整
                x_vals = np.linspace(-10, 10, 400)
                y_vals = f_lambdified(x_vals)

                plt.figure()
                plt.plot(x_vals, y_vals, label=str(function))
                plt.title('函数图像')
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.axhline(0, color='black', lw=0.5, ls='--')
                plt.axvline(0, color='black', lw=0.5, ls='--')
                plt.grid()
                plt.legend()
                plt.xlim(-10, 10)
                plt.ylim(-10, 10)

        # 将图形保存到内存中
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        graph = base64.b64encode(buf.read()).decode('utf-8')

    return render(request, 'calculator/calculus.html', {'result': result, 'graph': graph})