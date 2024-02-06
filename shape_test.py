import matplotlib.pyplot as plt 
import os

import numpy as np
import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

from mitsuba import ScalarTransform4f as T

from optimizer import *

import sys


if __name__ == "__main__":
    
    it_time = 150

    control = sys.argv[1]
    set_mode = sys.argv[2]

    control = int(control)
    set_mode = int(set_mode)

    legends = ['SO(3) Matrix','so(3)','Unit Quaternion 1','Unit Quaternion 2','Euler Angle','se(3)']
    opt = ['Adam','Uniform Adam','Vector Adam']

    print('Parameterization:' + legends[control] + 'Optimizer:'+ opt[set_mode])

    lr_rate_1 = 1e-1
    # lr_rate = 5e-2
    lr_rate_m = 1e-2
    lr_rate_v = 1e-2
    lr_rate_q = 1e-2

    dataset = 0
    sensor_count = 12

    golden_ratio = (1 + 5**0.5)/2

    disturb_coef = 0.5
    sensor_init = []
    sensor = []

    origin = np.load('origin.npy')
    rand_a = np.load('02020'+str(dataset)+'_rand_a.npy')
    rand_b = np.load('02020'+str(dataset)+'_rand_b.npy')
    for i in range(sensor_count):
        # theta = 2 * dr.pi * i / golden_ratio
        # phi = dr.acos(1 - 2*(i+0.5)/sensor_count)
        
        # d = 5
        # origin = [
        #     d * dr.cos(theta) * dr.sin(phi),
        #     d * dr.sin(theta) * dr.sin(phi),
        #     d * dr.cos(phi)
        # ]
        sensor_init.append({
            'type': 'perspective',
            'fov': 45,
            'to_world': T.look_at(target=[0, 0, 0], origin=origin[i].tolist(), up=[0, 1, 0]),
            'film': {
                'type': 'hdrfilm',
                'filter': {'type': 'gaussian'},
                'sample_border': True
            }
        })

        Trans = T.translate(disturb_coef * rand_a[i] * 1.2)
        disturb = rand_b[i]
        disturb_n = np.linalg.norm(disturb)
        disturb_v = disturb / disturb_n
        Rotate = T.rotate(disturb_v, disturb_coef * disturb_n * 90)

        
        sensor.append({
            'type': 'perspective',
            'fov': 45,
            'to_world': Trans @ Rotate @ T.look_at(target=[0, 0, 0], origin=origin[i].tolist(), up=[0, 1, 0]),
            'film': {
                'type': 'hdrfilm',
                'filter': {'type': 'gaussian'},
                'sample_border': True
            }
        })
    sensor_init = np.array(sensor_init)

    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'direct_reparam',
        },
        'sensor_0': sensor[0],
        'sensor_1': sensor[1],
        'sensor_2': sensor[2],
        'sensor_3': sensor[3],
        'sensor_4': sensor[4],
        'sensor_5': sensor[5],
        'sensor_6': sensor[6],
        'sensor_7': sensor[7],
        'sensor_8': sensor[8],
        'sensor_9': sensor[9],
        'sensor_10': sensor[10],
        'sensor_11': sensor[11],
        'emitter': {
            'type': 'envmap',
            'filename': "../scenes/textures/envmap2.exr",
        },
        'shape': {
            'type': 'ply',
            'filename': "../scenes/meshes/suzanne.ply",
            'bsdf': {'type': 'diffuse'}
        }
        # 'emitter': {
        #     'type': 'envmap',
        #     'filename': "../scenes/textures/envmap.exr",
        # },
        # 'shape': {
        #     'type': 'ply',
        #     'filename': "../scenes/meshes/bunny.ply",
        #     'to_world': T.scale(12),
        #     'bsdf': {'type': 'diffuse'}
        # }
    }

    scene_target = mi.load_dict(scene_dict)
    ref_spp = 256

    ref_img = [mi.render(scene_target, sensor=mi.load_dict(sensor_init[i]), spp=ref_spp) for i in range(sensor_count)]
    # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
    for i in range(sensor_count):
        mi.util.write_bitmap('./results/'+str(dataset)+'_reference'+str(i)+'.png', ref_img[i])
        # axs[i].imshow(mi.util.convert_to_bitmap(ref_img[i]))
        # axs[i].axis('off')
        # plt.imshow(mi.util.convert_to_bitmap(ref_img[i]))
        # plt.axis('off')
        # plt.savefig('./results/'+str(dataset)+'_reference'+str(i)+'.png')
    # plt.savefig('./results/'+str(dataset)+'_reference.png')


    scene_dict['shape']['filename'] = '../scenes/meshes/ico_10k.ply'
    # scene_dict['shape']['to_world'] = T.scale(1)
    scene_source = mi.load_dict(scene_dict)

    init_img = [mi.render(scene_source, sensor=i, spp=ref_spp) for i in range(sensor_count)]
    # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
    # for i in range(sensor_count):
    #     axs[i].imshow(mi.util.convert_to_bitmap(init_img[i]))
    #     axs[i].axis('off')
    # plt.savefig('./results/'+str(dataset)+'_initial.png')
    for i in range(sensor_count):
        mi.util.write_bitmap('./results/'+str(dataset)+'_initial'+str(i)+'.png', init_img[i])

    params = mi.traverse(scene_source)

    lambda_ = 25
    ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)


    ######################################################################################################
    ######################################### Original Adam ##############################################
    ######################################################################################################

    if control == 0:
        ######################################### Without ##############################################
        opt_1 = mi.ad.Adam(lr=lr_rate_1, uniform = True)
        opt_1['u'] = ls.to_differential(params['shape.vertex_positions'])

        iterations = it_time if 'PYTEST_CURRENT_TEST' not in os.environ else 5
        loss_hist0 = []
        for it in range(iterations):
            loss = mi.Float(0.0)

            # Retrieve the vertex positions from the latent variable
            params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
            params.update()

            loss = 0
            for i in range(sensor_count):
                img = mi.render(scene_source,params,sensor = mi.load_dict(sensor[i]), seed=it, spp=16)

                # L1 Loss
                loss += dr.mean(dr.abs(img - ref_img[i]))
            dr.backward(loss)
            loss_hist0.append(loss)
            opt_1.step()

            print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')

        plt.figure()
        plt.plot(loss_hist0)
        plt.savefig('./results/'+str(dataset)+'_Adam_without_loss.png')
        np.save('./results/'+str(dataset)+'_Adam_without_loss.npy',loss_hist0)

        # Update the mesh after the last iteration's gradient step
        params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
        params.update();

        final_img = [mi.render(scene_source, params,sensor = mi.load_dict(sensor[i]), spp=128) for i in range(sensor_count)]

        # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
        # for i in range(sensor_count):
        #     axs[i].imshow(mi.util.convert_to_bitmap(final_img[i]))
        #     axs[i].axis('off')
        # plt.savefig('./results/'+str(dataset)+'_Adam'+'_without.png')
        for i in range(sensor_count):
            mi.util.write_bitmap('./results/'+str(dataset)+'_Adam'+'_without'+str(i)+'.png', final_img[i])

    if control == 1:
        ######################################### Matrix ##############################################
        def apply_transformation(params, opt,i):
            
            trafo = mi.Transform4f.translate(opt['T_'+str(i)])  @ mi.Matrix4f(opt['R_'+str(i)])
            
            params['sensor_'+str(i)+'.to_world'] = trafo
            params.update()

        scene_source = mi.load_dict(scene_dict)
        params = mi.traverse(scene_source)

        lambda_ = 25
        ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
        opt_1 = mi.ad.Adam(lr=lr_rate_1, uniform = True)
        opt_1['u'] = ls.to_differential(params['shape.vertex_positions'])

        opt = R_Adam(lr=lr_rate_m, mode=set_mode)
        for i in range(sensor_count):
            opt['R_'+str(i)] = mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix)
            opt['T_'+str(i)] = params['sensor_'+str(i)+'.to_world'].translation()

        iterations = it_time if 'PYTEST_CURRENT_TEST' not in os.environ else 5

        loss_hist = []
        loss_transform = []
        loss_t = []
        loss_r = []
        loss_ts = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        loss_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_ps = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        for it in range(iterations):
            loss = mi.Float(0.0)

            # Retrieve the vertex positions from the latent variables
            params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
            params.update()

            with dr.suspend_grad():
                for i in range(sensor_count):
                    record_rs[i].append(opt['R_'+str(i)])
                    record_ps[i].append([opt['T_'+str(i)].x[0],opt['T_'+str(i)].y[0],opt['T_'+str(i)].z[0]])

            for i in range(sensor_count):
                apply_transformation(params, opt,i)
            
            loss = 0
            l = 0
            lt = 0
            lr = 0
            for i in range(sensor_count):
                with dr.suspend_grad():
                    l += dr.sum(dr.sum(dr.unravel(mi.Matrix4f, dr.sqr(dr.ravel(mi.Matrix4f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    lt += dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation()))
                    lr += dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    loss_ts[i].append(dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation())))
                    loss_rs[i].append(dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix)))))))
                img = mi.render(scene_source,params, sensor=i,seed=it, spp=16)

                # L1 Loss
                loss += dr.mean(dr.abs(img - ref_img[i]))
            dr.backward(loss)
            loss_hist.append(loss)
            loss_transform.append(l)
            loss_t.append(lt)
            loss_r.append(lr)

            opt_1.step()
            opt.step()

            print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')

        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_rs'+str(i)+'.npy',record_rs[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_ps'+str(i)+'.npy',record_ps[i])


        plt.figure()
        plt.plot(loss_hist)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_loss.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_loss.npy',loss_hist)


        plt.figure()
        plt.plot(loss_transform)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_transform_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_transform_error.npy',loss_transform)


        plt.figure()
        plt.plot(loss_t)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_translation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_translation_error.npy',loss_t)


        plt.figure()
        plt.plot(loss_r)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_rotation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix_rotation_error.npy',loss_r)

        name = 'matrix'
        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_t'+str(i)+'.npy',loss_ts[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_r'+str(i)+'.npy',loss_rs[i])

        # Update the mesh after the last iteration's gradient step
        params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
        params.update();

        final_img = [mi.render(scene_source, params,sensor = i, spp=128) for i in range(sensor_count)]

        # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
        # for i in range(sensor_count):
        #     axs[i].imshow(mi.util.convert_to_bitmap(final_img[i]))
        #     axs[i].axis('off')
        # plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix.png')
        # plt.close()
        for i in range(sensor_count):
            mi.util.write_bitmap('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_matrix'+str(i)+'.png', final_img[i])

    if control == 2:
        ######################################### Vector ##############################################
        thresh = 0.01
        def SO3_to_so3(R,eps=1e-7):
            # print(R)
            # trace = dr.trace(R)
            trace = R[0][0] + R[1][1] + R[2][2]
            # print(R)
            theta = dr.acos(dr.clamp((trace - 1) / 2, -1 + eps, 1 - eps))[0] % dr.pi # ln(R) will explode if theta==pi
            if theta < thresh:
                A = taylor_A(theta)
            else:
                A = dr.sin(theta) / theta
            lnR = 1/(2*A+1e-8)*(R-dr.transpose(R))
            w0,w1,w2 = lnR[2,1],lnR[0,2],lnR[1,0]
            w = mi.Point3f(w0[0],w1[0],w2[0])
            # print(R)
            return w

        def apply_transformation(params, opt,i):
            
            trafo = mi.Transform4f.translate(opt['T_'+str(i)])  @ mi.Matrix4f(so3_to_SO3(w_to_wx(opt['R_'+str(i)])))
            
            params['sensor_'+str(i)+'.to_world'] = trafo
            params.update()

        scene_source = mi.load_dict(scene_dict)
        params = mi.traverse(scene_source)

        lambda_ = 25
        ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
        opt_1 = mi.ad.Adam(lr=lr_rate_1, uniform = True)
        opt_1['u'] = ls.to_differential(params['shape.vertex_positions'])

        opt = R_Adam(lr=lr_rate_v, mode=set_mode)
        for i in range(sensor_count):
            opt['R_'+str(i)] = SO3_to_so3(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix))
            opt['T_'+str(i)] = params['sensor_'+str(i)+'.to_world'].translation()

        iterations = it_time if 'PYTEST_CURRENT_TEST' not in os.environ else 5

        loss_hist = []
        loss_transform = []
        loss_t = []
        loss_r = []
        loss_ts = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        loss_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_ps = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        for it in range(iterations):
            loss = mi.Float(0.0)

            # Retrieve the vertex positions from the latent variables
            params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
            params.update()

            with dr.suspend_grad():
                for i in range(sensor_count):
                    record_rs[i].append(so3_to_SO3(w_to_wx(opt['R_'+str(i)])))
                    record_ps[i].append([opt['T_'+str(i)].x[0],opt['T_'+str(i)].y[0],opt['T_'+str(i)].z[0]])

            for i in range(sensor_count):
                apply_transformation(params, opt,i)
            
            loss = 0
            l = 0
            lt = 0
            lr = 0
            for i in range(sensor_count):
                with dr.suspend_grad():
                    l += dr.sum(dr.sum(dr.unravel(mi.Matrix4f, dr.sqr(dr.ravel(mi.Matrix4f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    lt += dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation()))
                    lr += dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    loss_ts[i].append(dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation())))
                    loss_rs[i].append(dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix)))))))
                img = mi.render(scene_source,params, sensor=i,seed=it, spp=16)

                # L1 Loss
                loss += dr.mean(dr.abs(img - ref_img[i]))
            dr.backward(loss)
            loss_hist.append(loss)
            loss_transform.append(l)
            loss_t.append(lt)
            loss_r.append(lr)

            opt_1.step()
            opt.step()

            print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')

        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_rs'+str(i)+'.npy',record_rs[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_ps'+str(i)+'.npy',record_ps[i])

        plt.figure()
        plt.plot(loss_hist)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_loss.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_loss.npy',loss_hist)

        plt.figure()
        plt.plot(loss_transform)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_transform_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_transform_error.npy',loss_transform)

        plt.figure()
        plt.plot(loss_t)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_translation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_translation_error.npy',loss_t)

        plt.figure()
        plt.plot(loss_r)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_rotation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector_rotation_error.npy',loss_r)

        name = 'vector'
        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_t'+str(i)+'.npy',loss_ts[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_r'+str(i)+'.npy',loss_rs[i])

        # Update the mesh after the last iteration's gradient step
        params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
        params.update();

        final_img = [mi.render(scene_source, params,sensor = i, spp=128) for i in range(sensor_count)]

        # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
        # for i in range(sensor_count):
        #     axs[i].imshow(mi.util.convert_to_bitmap(final_img[i]))
        #     axs[i].axis('off')
        # plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector.png')
        # plt.close()
        for i in range(sensor_count):
            mi.util.write_bitmap('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_vector'+str(i)+'.png', final_img[i])


    if control == 3:
        ######################################### Quaternion ##############################################

        def apply_transformation(params, opt,i):

            trafo = mi.Transform4f.translate(opt['T_'+str(i)])  @ dr.quat_to_matrix(mi.Quaternion4f(opt['q_'+str(i)]))
            
            params['sensor_'+str(i)+'.to_world'] = trafo
            params.update()

        scene_source = mi.load_dict(scene_dict)
        params = mi.traverse(scene_source)

        lambda_ = 25
        ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
        opt_1 = mi.ad.Adam(lr=lr_rate_1, uniform = True)
        opt_1['u'] = ls.to_differential(params['shape.vertex_positions'])

        opt = R_Adam(lr=lr_rate_q, mode=set_mode)
        for i in range(sensor_count):
            opt['q_'+str(i)] = mi.Vector4f(dr.matrix_to_quat(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix)))
            opt['T_'+str(i)] = params['sensor_'+str(i)+'.to_world'].translation()

        iterations = it_time if 'PYTEST_CURRENT_TEST' not in os.environ else 5

        loss_hist = []
        loss_transform = []
        loss_t = []
        loss_r = []
        loss_ts = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        loss_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_ps = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        for it in range(iterations):
            loss = mi.Float(0.0)

            # Retrieve the vertex positions from the latent variables
            params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
            params.update()

            with dr.suspend_grad():
                for i in range(sensor_count):
                    record_rs[i].append(mi.Matrix3f(dr.quat_to_matrix(mi.Quaternion4f(opt['q_'+str(i)]))))
                    record_ps[i].append([opt['T_'+str(i)].x[0],opt['T_'+str(i)].y[0],opt['T_'+str(i)].z[0]])

            for i in range(sensor_count):
                opt['q_'+str(i)] = dr.normalize(opt['q_'+str(i)])
                apply_transformation(params, opt,i)
            
            loss = 0
            l = 0
            lt = 0
            lr = 0
            for i in range(sensor_count):
                with dr.suspend_grad():
                    l += dr.sum(dr.sum(dr.unravel(mi.Matrix4f, dr.sqr(dr.ravel(mi.Matrix4f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    lt += dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation()))
                    lr += dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    loss_ts[i].append(dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation())))
                    loss_rs[i].append(dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix)))))))
                img = mi.render(scene_source,params, sensor=i,seed=it, spp=16)

                # L1 Loss
                loss += dr.mean(dr.abs(img - ref_img[i]))
            dr.backward(loss)
            loss_hist.append(loss)
            loss_transform.append(l)
            loss_t.append(lt)
            loss_r.append(lr)

            opt_1.step()
            opt.step()

            print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')

        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_1_quaternion_rs'+str(i)+'.npy',record_rs[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_1_quaternion_ps'+str(i)+'.npy',record_ps[i])


        plt.figure()
        plt.plot(loss_hist)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion_loss.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion_loss.npy',loss_hist)

        plt.figure()
        plt.plot(loss_transform)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion_transform_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion_transform_error.npy',loss_transform)

        plt.figure()
        plt.plot(loss_t)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion_translation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion_translation_error.npy',loss_t)

        plt.figure()
        plt.plot(loss_r)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion_rotation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion_rotation_error.npy',loss_r)

        name = 'quaternion'
        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_t'+str(i)+'.npy',loss_ts[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_r'+str(i)+'.npy',loss_rs[i])


        # Update the mesh after the last iteration's gradient step
        params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
        params.update();

        final_img = [mi.render(scene_source, params,sensor = i, spp=128) for i in range(sensor_count)]

        # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
        # for i in range(sensor_count):
        #     axs[i].imshow(mi.util.convert_to_bitmap(final_img[i]))
        #     axs[i].axis('off')
        # plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion.png')
        # plt.close()
        for i in range(sensor_count):
            mi.util.write_bitmap('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion1'+str(i)+'.png', final_img[i])
        

    if control == 4:
        ######################################### Quaternion2 ##############################################

        def apply_transformation(params, opt,i):

            trafo = mi.Transform4f.translate(opt['T_'+str(i)])  @ dr.quat_to_matrix(mi.Quaternion4f(opt['q_'+str(i)]))
            
            params['sensor_'+str(i)+'.to_world'] = trafo
            params.update()

        scene_source = mi.load_dict(scene_dict)
        params = mi.traverse(scene_source)

        lambda_ = 25
        ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
        opt_1 = mi.ad.Adam(lr=lr_rate_1, uniform = True)
        opt_1['u'] = ls.to_differential(params['shape.vertex_positions'])

        opt = Q_Adam(lr=lr_rate_q, mode=set_mode)
        for i in range(sensor_count):
            opt['q_'+str(i)] = mi.Vector4f(dr.matrix_to_quat(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix)))
            opt['T_'+str(i)] = params['sensor_'+str(i)+'.to_world'].translation()

        iterations = it_time if 'PYTEST_CURRENT_TEST' not in os.environ else 5

        loss_hist = []
        loss_transform = []
        loss_t = []
        loss_r = []
        loss_ts = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        loss_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_ps = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        for it in range(iterations):
            loss = mi.Float(0.0)

            # Retrieve the vertex positions from the latent variables
            params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
            params.update()

            with dr.suspend_grad():
                for i in range(sensor_count):
                    record_rs[i].append(mi.Matrix3f(dr.quat_to_matrix(mi.Quaternion4f(opt['q_'+str(i)]))))
                    record_ps[i].append([opt['T_'+str(i)].x[0],opt['T_'+str(i)].y[0],opt['T_'+str(i)].z[0]])

            for i in range(sensor_count):
                apply_transformation(params, opt,i)
            
            loss = 0
            l = 0
            lt = 0
            lr = 0
            for i in range(sensor_count):
                with dr.suspend_grad():
                    l += dr.sum(dr.sum(dr.unravel(mi.Matrix4f, dr.sqr(dr.ravel(mi.Matrix4f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    lt += dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation()))
                    lr += dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    loss_ts[i].append(dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation())))
                    loss_rs[i].append(dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix)))))))
                img = mi.render(scene_source,params, sensor=i,seed=it, spp=16)

                # L1 Loss
                loss += dr.mean(dr.abs(img - ref_img[i]))
            dr.backward(loss)
            loss_hist.append(loss)
            loss_transform.append(l)
            loss_t.append(lt)
            loss_r.append(lr)

            opt_1.step()
            opt.step()

            print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')
        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_rs'+str(i)+'.npy',record_rs[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_ps'+str(i)+'.npy',record_ps[i])


        plt.figure()
        plt.plot(loss_hist)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_loss.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_loss.npy',loss_hist)

        plt.figure()
        plt.plot(loss_transform)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_transform_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_transform_error.npy',loss_transform)

        plt.figure()
        plt.plot(loss_t)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_translation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_translation_error.npy',loss_t)

        plt.figure()
        plt.plot(loss_r)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_rotation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion_rotation_error.npy',loss_r)

        name = '2_quaternion'
        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_t'+str(i)+'.npy',loss_ts[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_r'+str(i)+'.npy',loss_rs[i])


        # Update the mesh after the last iteration's gradient step
        params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
        params.update();

        final_img = [mi.render(scene_source, params,sensor = i, spp=128) for i in range(sensor_count)]

        # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
        # for i in range(sensor_count):
        #     axs[i].imshow(mi.util.convert_to_bitmap(final_img[i]))
        #     axs[i].axis('off')
        # plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_2_quaternion.png')
        # plt.close()
        for i in range(sensor_count):
            mi.util.write_bitmap('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_quaternion2'+str(i)+'.png', final_img[i])

        test_img = [mi.render(scene_source, sensor=mi.load_dict(sensor_init[i]), spp=ref_spp) for i in range(sensor_count)]
        # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))

        scene_dict['shape']['filename'] = '../scenes/meshes/ico_10k.ply'
        scene_source = mi.load_dict(scene_dict)
        test_img2 = [mi.render(scene_source, sensor=mi.load_dict(sensor_init[i]), spp=ref_spp) for i in range(sensor_count)]
        for i in range(sensor_count):
            mi.util.write_bitmap('./results/'+str(dataset)+'test1'+str(i)+'.png', test_img[i])
            mi.util.write_bitmap('./results/'+str(dataset)+'test2'+str(i)+'.png', test_img2[i])
    if control == 5:
        ######################################### Euler Angles ##############################################

        def apply_transformation(params, opt,i):

            trafo = mi.Transform4f.translate(opt['T_'+str(i)]).rotate([0, 0, 1], opt['R_'+str(i)].z ).rotate([0, 1, 0], opt['R_'+str(i)].y).rotate([1, 0, 0],opt['R_'+str(i)].x)
            
            params['sensor_'+str(i)+'.to_world'] = trafo
            params.update()

        scene_source = mi.load_dict(scene_dict)
        params = mi.traverse(scene_source)

        lambda_ = 25
        ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
        opt_1 = mi.ad.Adam(lr=lr_rate_1, uniform = True)
        opt_1['u'] = ls.to_differential(params['shape.vertex_positions'])

        opt = R_Adam(lr=lr_rate_q, mode=set_mode)

        for i in range(sensor_count):
            euler = dr.quat_to_euler(dr.matrix_to_quat(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix)))
            opt['R_'+str(i)] = mi.Vector3f(np.rad2deg(euler.x),np.rad2deg(euler.y),np.rad2deg(euler.z))
            opt['T_'+str(i)] = params['sensor_'+str(i)+'.to_world'].translation()

        iterations = it_time if 'PYTEST_CURRENT_TEST' not in os.environ else 5

        loss_hist = []
        loss_transform = []
        loss_t = []
        loss_r = []
        # loss_ts = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
        # loss_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
        loss_ts = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        loss_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_ps = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        for it in range(iterations):
            loss = mi.Float(0.0)

            # Retrieve the vertex positions from the latent variables
            params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
            params.update()

            with dr.suspend_grad():
                for i in range(sensor_count):
                    record_rs[i].append(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix))
                    record_ps[i].append([opt['T_'+str(i)].x[0],opt['T_'+str(i)].y[0],opt['T_'+str(i)].z[0]])

            for i in range(sensor_count):
                apply_transformation(params, opt,i)
            
            loss = 0
            l = 0
            lt = 0
            lr = 0
            for i in range(sensor_count):
                with dr.suspend_grad():
                    l += dr.sum(dr.sum(dr.unravel(mi.Matrix4f, dr.sqr(dr.ravel(mi.Matrix4f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    lt += dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation()))
                    lr += dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    loss_ts[i].append(dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation())))
                    loss_rs[i].append(dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix)))))))
                img = mi.render(scene_source,params, sensor=i,seed=it, spp=16)

                # L1 Loss
                loss += dr.mean(dr.abs(img - ref_img[i]))
            dr.backward(loss)
            loss_hist.append(loss)
            loss_transform.append(l)
            loss_t.append(lt)
            loss_r.append(lr)

            opt_1.step()
            opt.step()

            print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')

        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_rs'+str(i)+'.npy',record_rs[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_ps'+str(i)+'.npy',record_ps[i])

        plt.figure()
        plt.plot(loss_hist)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_loss.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_loss.npy',loss_hist)

        plt.figure()
        plt.plot(loss_transform)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_transform_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_transform_error.npy',loss_transform)

        plt.figure()
        plt.plot(loss_t)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_translation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_translation_error.npy',loss_t)

        plt.figure()
        plt.plot(loss_r)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_rotation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler_rotation_error.npy',loss_r)

        name = 'euler'
        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_t'+str(i)+'.npy',loss_ts[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_r'+str(i)+'.npy',loss_rs[i])


        # Update the mesh after the last iteration's gradient step
        params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
        params.update();

        final_img = [mi.render(scene_source, params,sensor = i, spp=128) for i in range(sensor_count)]

        # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
        # for i in range(sensor_count):
        #     axs[i].imshow(mi.util.convert_to_bitmap(final_img[i]))
        #     axs[i].axis('off')
        # plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler.png')
        # plt.close()
        for i in range(sensor_count):
            mi.util.write_bitmap('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_euler'+str(i)+'.png', final_img[i])
        

    if control == 6:
        ######################################### SE3 ##############################################

        def apply_transformation(params, opt,i):

            trafo = se3_to_SE3(opt['w_'+str(i)],opt['u_'+str(i)])
            
            params['sensor_'+str(i)+'.to_world'] = trafo
            params.update()

        scene_source = mi.load_dict(scene_dict)
        params = mi.traverse(scene_source)

        lambda_ = 25
        ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
        opt_1 = mi.ad.Adam(lr=lr_rate_1, uniform = True)
        opt_1['u'] = ls.to_differential(params['shape.vertex_positions'])

        opt = R_Adam(lr=lr_rate_q, mode=set_mode)

        for i in range(sensor_count):
            wu = SE3_to_se3(params['sensor_'+str(i)+'.to_world'].matrix)
            opt['w_'+str(i)] = mi.Point3f(wu[0],wu[1],wu[2])
            opt['u_'+str(i)] = mi.Point3f(wu[3],wu[4],wu[5])

        iterations = it_time if 'PYTEST_CURRENT_TEST' not in os.environ else 5

        loss_hist = []
        loss_transform = []
        loss_t = []
        loss_r = []
        loss_ts = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        loss_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_rs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        record_ps = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        for it in range(iterations):
            loss = mi.Float(0.0)

            # Retrieve the vertex positions from the latent variables
            params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
            params.update()

            with dr.suspend_grad():
                for i in range(sensor_count):
                    record_rs[i].append(mi.Matrix3f(se3_to_SE3(opt['w_'+str(i)],opt['u_'+str(i)]).matrix))
                    record_ps[i].append([se3_to_SE3(opt['w_'+str(i)],opt['u_'+str(i)]).translation().x[0],se3_to_SE3(opt['w_'+str(i)],opt['u_'+str(i)]).translation().y[0],se3_to_SE3(opt['w_'+str(i)],opt['u_'+str(i)]).translation().z[0]])

            for i in range(sensor_count):
                apply_transformation(params, opt,i)
            
            loss = 0
            l = 0
            lt = 0
            lr = 0
            for i in range(sensor_count):
                with dr.suspend_grad():
                    l += dr.sum(dr.sum(dr.unravel(mi.Matrix4f, dr.sqr(dr.ravel(mi.Matrix4f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    lt += dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation()))
                    lr += dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix))))))
                    loss_ts[i].append(dr.sum(dr.sqr(params['sensor_'+str(i)+'.to_world'].translation() - sensor_init[i]['to_world'].translation())))
                    loss_rs[i].append(dr.sum(dr.sum(dr.unravel(mi.Matrix3f, dr.sqr(dr.ravel(mi.Matrix3f(params['sensor_'+str(i)+'.to_world'].matrix - sensor_init[i]['to_world'].matrix)))))))
                img = mi.render(scene_source,params, sensor=i,seed=it, spp=16)

                # L1 Loss
                loss += dr.mean(dr.abs(img - ref_img[i]))
            dr.backward(loss)
            loss_hist.append(loss)
            loss_transform.append(l)
            loss_t.append(lt)
            loss_r.append(lr)

            opt_1.step()
            opt.step()

            print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}", end='\r')

        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_rs'+str(i)+'.npy',record_rs[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_ps'+str(i)+'.npy',record_ps[i])


        plt.figure()
        plt.plot(loss_hist)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_loss.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_loss.npy',loss_hist)

        plt.figure()
        plt.plot(loss_transform)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_transform_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_transform_error.npy',loss_transform)

        plt.figure()
        plt.plot(loss_t)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_translation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_translation_error.npy',loss_t)

        plt.figure()
        plt.plot(loss_r)
        plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_rotation_error.png')
        np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se_rotation_error.npy',loss_r)

        name = 'se3'
        for i in range(sensor_count):
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_t'+str(i)+'.npy',loss_ts[i])
            np.save('./results/'+str(dataset)+'_Adam'+str(set_mode)+name+'_loss_r'+str(i)+'.npy',loss_rs[i])

        # Update the mesh after the last iteration's gradient step
        params['shape.vertex_positions'] = ls.from_differential(opt_1['u'])
        params.update();

        final_img = [mi.render(scene_source, params,sensor = i, spp=128) for i in range(sensor_count)]

        # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
        # for i in range(sensor_count):
        #     axs[i].imshow(mi.util.convert_to_bitmap(final_img[i]))
        #     axs[i].axis('off')
        # plt.savefig('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se.png')
        # plt.close()
        for i in range(sensor_count):
            mi.util.write_bitmap('./results/'+str(dataset)+'_Adam'+str(set_mode)+'_se'+str(i)+'.png', final_img[i])
        


    vertices = np.array(params['shape.vertex_positions'])
    vertices = np.reshape(vertices, (-1,3))


    np.save('./results/meshes/'+str(dataset)+'_Adam'+str(set_mode)+'_vertices_'+str(control)+'.npy',vertices)