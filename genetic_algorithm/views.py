import os
from rest_framework import views, status, response
from genetic_algorithm import solution


# Create your views here.
class Solve(views.APIView):
    def post(self, request):
        data = solution.main(request.data['num_of_vehicles'], request.data['atm_list'])
        return response.Response({'response': 'Success', 'data': data}, status=status.HTTP_200_OK)